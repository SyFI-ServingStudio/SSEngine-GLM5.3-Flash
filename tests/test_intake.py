import msgspec
import numpy as np
import pytest

from ppimpl.serve.intake import RawRequest, parse_ids


def ids_of(body: bytes) -> np.ndarray:
    return parse_ids(msgspec.json.decode(body, type=RawRequest).token_ids)


def test_parse_ids():
    assert ids_of(b'{"token_ids": [1, 2,3 , 151000]}').tolist() == [1, 2, 3, 151000]
    assert ids_of(b'{"token_ids": [ ]}').dtype == np.int32 and len(ids_of(b'{"token_ids": []}')) == 0
    for bad in (b'[1.5, 2]', b'["a"]', b'[1, -2]', b'[1, 2147483648]', b'7', b'[1,,2]', b'[[1]]'):
        with pytest.raises(ValueError):
            ids_of(b'{"token_ids": ' + bad + b'}')


def test_fast_path_equals_the_old_one():
    """ppimpl/serve/ids.py: the same ids as msgspec + numpy, for long arrays (split over threads) and short ones,
    and every body it does not take goes to the old path with the old answer or error."""
    from ppimpl.serve import ids as ids_mod
    from ppimpl.serve.intake import Intake, parse_body

    rng = np.random.default_rng(0)
    it = Intake.__new__(Intake)
    Intake.__init__(it, None, 0)
    for n in (1, 7, 1000, 525312):
        ids = rng.integers(0, 154880, n)
        body = msgspec.json.encode({"model": "m", "rid": "r", "token_ids": ids.tolist(),
                                    "sampling_params": {"max_tokens": 1, "temperature": 0, "ignore_eos": True},
                                    "stream": True, "stream_options": {"include_usage": True}})
        got = ids_mod.parse_fast(body, it.rest_dec)
        assert got is not None, n
        req, parsed = got
        assert parsed.dtype == np.int32 and parsed.tolist() == ids.tolist()
        assert (req.model, req.rid, req.stream, req.sampling_params.max_tokens) == ("m", "r", True, 1)
    spaced = b'{"token_ids" : [ 1 ,\n2,3 , 151000 ] , "stream": false}'
    assert ids_mod.parse_fast(spaced, it.rest_dec)[1].tolist() == [1, 2, 3, 151000]
    # not taken by the fast path, same result as before
    odd = [b'{"model": "\\"token_ids\\": [9]", "token_ids": [1, 2]}',      # the key inside a string
           b'{"sampling_params": {"max_tokens": 1}, "token_ids": [1, 2], "token_ids": [3]}',   # twice
           b'{"token_ids": [1, 2], "x": 1}']                               # unknown field: allowed by both
    for body in odd:
        slow = ids_of(body)
        assert parse_body(body, it.dec, it.rest_dec)[1].tolist() == slow.tolist()
    for bad in (b'[1.5, 2]', b'["a"]', b'[1, -2]', b'[1, 2147483648]', b'7', b'[1,,2]', b'[[1]]', b'[1 2]',
                b'[01, 2]'):
        body = b'{"token_ids": ' + bad + b'}'
        assert ids_mod.parse_fast(body, it.rest_dec) is None
        with pytest.raises((ValueError, msgspec.DecodeError, msgspec.ValidationError)):
            parse_body(body, it.dec, it.rest_dec)
