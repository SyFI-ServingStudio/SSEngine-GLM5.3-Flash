import msgspec
import numpy as np
import pytest

from ssengine.serve.intake import RawRequest, parse_ids


def ids_of(body: bytes) -> np.ndarray:
    return parse_ids(msgspec.json.decode(body, type=RawRequest).token_ids)


def test_parse_ids():
    assert ids_of(b'{"token_ids": [1, 2,3 , 151000]}').tolist() == [1, 2, 3, 151000]
    assert ids_of(b'{"token_ids": [ ]}').dtype == np.int32 and len(ids_of(b'{"token_ids": []}')) == 0
    for bad in (b'[1.5, 2]', b'["a"]', b'[1, -2]', b'[1, 2147483648]', b'7', b'[1,,2]', b'[[1]]'):
        with pytest.raises(ValueError):
            ids_of(b'{"token_ids": ' + bad + b'}')


def test_fast_path_equals_the_old_one():
    """ssengine/serve/ids.py: the same ids as msgspec + numpy, for long arrays (split over threads) and short ones,
    and every body it does not take goes to the old path with the old answer or error."""
    from ssengine.serve import ids as ids_mod
    from ssengine.serve.intake import Intake, parse_body

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


def test_stream_parser_equals_the_fast_path():
    """ids.StreamParser (the ids parsed while the body arrives) gives parse_fast's answer for any chunking, and gives
    up (None: the caller parses the whole body) on every body parse_fast does not take."""
    from ssengine.serve import ids as ids_mod
    from ssengine.serve.intake import Intake

    it = Intake.__new__(Intake)
    Intake.__init__(it, None, 0)

    def stream(body, size, alloc=None):
        sp = ids_mod.StreamParser(it.rest_dec)
        for i in range(0, len(body), size):
            sp.feed(body[i:i + size])
        got = sp.finish(alloc)
        assert sp.body() == body
        return got

    rng = np.random.default_rng(1)
    for n in (1, 7, 1000, 525312):
        ids = rng.integers(0, 154880, n)
        body = msgspec.json.encode({"model": "m", "rid": "r", "token_ids": ids.tolist(),
                                    "sampling_params": {"max_tokens": 1}, "stream": True})
        for size in (1, 5, 4096, 65536, 1 << 30):
            if size < 64 and n > 1000:
                continue
            req, parsed, tok = stream(body, size)
            assert tok is None and parsed.dtype == np.int32 and parsed.tolist() == ids.tolist(), (n, size)
            assert (req.model, req.rid, req.stream) == ("m", "r", True)
        buf = np.zeros(n + 3, np.int32)
        req, parsed, tok = stream(body, 65536, lambda k: ("at", buf[:k]))
        assert tok == "at" and np.shares_memory(parsed, buf) and buf[:n].tolist() == ids.tolist()
    spaced = b'{"token_ids" : [ 1 ,\n2,3 , 151000 ] , "stream": false}'
    for size in (1, 2, 3, 100):
        assert stream(spaced, size)[1].tolist() == [1, 2, 3, 151000]
    odd = [b'{"model": "\\"token_ids\\": [9]", "token_ids": [1, 2]}',
           b'{"sampling_params": {"max_tokens": 1}, "token_ids": [1, 2], "token_ids": [3]}',
           b'{"token_ids": []}', b'{"model": "token_ids", "token_ids": [1]}', b'{"x": 1, "token_ids": [1, 2]']
    bad = [b'[1.5, 2]', b'["a"]', b'[1, -2]', b'[1, 2147483648]', b'7', b'[1,,2]', b'[[1]]', b'[1 2]', b'[01, 2]']
    for body in odd:            # given up, or the old path's answer
        for size in (1, 3, 1 << 20):
            got = stream(body, size)
            assert got is None or got[1].tolist() == ids_of(body).tolist(), (body, size)
    for body in [b'{"token_ids": []}', b'{"x": 1, "token_ids": [1, 2]'] + [b'{"token_ids": ' + b + b'}' for b in bad]:
        for size in (1, 3, 1 << 20):
            assert stream(body, size) is None, (body, size)
    assert stream(b'{"token_ids": [1, 2], "x": 1}', 2)[1].tolist() == [1, 2]
