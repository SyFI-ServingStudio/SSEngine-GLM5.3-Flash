"""CPU tests of the FAST switch plumbing (lane m5-glue)."""

import pytest


def test_fast_from_env_sets_known_switches_only():
    from ppimpl.model import pp_patch

    saved = dict(pp_patch.FAST)
    try:
        pp_patch.fast_from_env("edge_send_eager=0,kpool_expand=0")
        assert pp_patch.FAST["edge_send_eager"] is False and pp_patch.FAST["kpool_expand"] is False
        pp_patch.fast_from_env("edge_send_eager=1")
        assert pp_patch.FAST["edge_send_eager"] is True
        pp_patch.fast_from_env("")
        assert pp_patch.FAST["edge_send_eager"] is True           # an empty value changes nothing
        with pytest.raises(AssertionError):
            pp_patch.fast_from_env("no_such_switch=1")
        with pytest.raises(AssertionError):
            pp_patch.fast_from_env("edge_send_eager=yes")
    finally:
        pp_patch.FAST.clear()
        pp_patch.FAST.update(saved)


def test_indexer_float_installs_and_uninstalls():
    """`install_indexer_float` finds the fork's `hidden_states.float()` line once, rebinds `Indexer.forward` to the
    edited source (a fork change fails here and at start-up), and `uninstall_indexer_float` restores the original."""
    from vllm.models.glm5next.common import attention

    from ppimpl.model import glue_fast

    orig = attention.Indexer.forward
    if glue_fast._ORIG_INDEXER_FORWARD is not None:
        orig = glue_fast._ORIG_INDEXER_FORWARD                   # installed earlier in this process
    glue_fast.install_indexer_float()
    try:
        new = attention.Indexer.forward
        assert new is not orig and new.__globals__["_pp_to_float"] is glue_fast.to_float
        assert new.__code__.co_filename.startswith("<glue_fast:")
    finally:
        glue_fast.uninstall_indexer_float()
    assert attention.Indexer.forward is orig


def test_to_float_cpu_falls_back_to_torch():
    import torch

    from ppimpl.model import glue_fast

    x = torch.randn(5, 7).to(torch.bfloat16)
    assert torch.equal(glue_fast.to_float(x), x.float())
