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
