"""CPU tests of the FAST switch plumbing (lane m5-glue)."""

import pytest


def test_fast_from_env_sets_known_switches_only():
    from ppimpl.model import pp_patch

    saved = dict(pp_patch.FAST)
    try:
        assert pp_patch.FAST["edge_send_eager"] is False          # off by default
        pp_patch.fast_from_env("edge_send_eager=1,kpool_expand=0")
        assert pp_patch.FAST["edge_send_eager"] is True and pp_patch.FAST["kpool_expand"] is False
        pp_patch.fast_from_env("")
        assert pp_patch.FAST["edge_send_eager"] is True           # an empty value changes nothing
        with pytest.raises(AssertionError):
            pp_patch.fast_from_env("no_such_switch=1")
        with pytest.raises(AssertionError):
            pp_patch.fast_from_env("edge_send_eager=yes")
    finally:
        pp_patch.FAST.clear()
        pp_patch.FAST.update(saved)
