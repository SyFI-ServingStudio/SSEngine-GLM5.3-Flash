"""CPU tests of the host trims (ppimpl/model/pp_patch.py): plain parameters keep storage and attributes; the
FlashInfer tactic memo returns what the autotuner returns, with the caller's runner object."""

import torch

from ppimpl.model import pp_patch


def test_plain_parameters_same_storage_and_attrs(monkeypatch):
    import vllm.model_executor.parameter as vp
    from vllm.model_executor.parameter import ModelWeightParameter

    monkeypatch.setattr(vp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vp, "get_tensor_model_parallel_world_size", lambda: 1)

    lin = torch.nn.Linear(8, 4, bias=False)
    w = ModelWeightParameter(data=torch.randn(4, 8), input_dim=1, output_dim=0, weight_loader=lambda *a: None)
    lin.weight = w
    assert pp_patch.plain_parameters([lin]) == 1
    assert type(lin.weight) is torch.nn.Parameter
    assert lin.weight.data_ptr() == w.data_ptr()
    assert lin.weight._output_dim == 0 and not lin.weight.requires_grad
    x = torch.randn(3, 8)
    assert torch.equal(lin(x), x @ w.data.t())
    assert pp_patch.plain_parameters([lin]) == 0


def test_autotune_memo_matches_and_returns_callers_runner():
    from flashinfer.autotuner.autotuner import AutoTuner, TunableRunner, TuningConfig

    class R(TunableRunner):
        def __init__(self, k):
            self.k = k

        def get_valid_tactics(self, inputs, profile, **kw):
            return [0]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kw):
            return inputs[0]

    pp_patch.install_autotune_memo()
    pp_patch.clear_autotune_memo()
    tuner = AutoTuner.get()
    cfg = TuningConfig()
    x = torch.zeros(16, 8)
    r1 = [R(1), R(2)]
    a = tuner.choose_one("ppimpl_test_op", r1, cfg, [x])
    r2 = [R(1), R(2)]
    b = tuner.choose_one("ppimpl_test_op", r2, cfg, [x])
    assert a[1] == b[1] and b[0] is r2[r1.index(a[0])]
    assert pp_patch.AUTOTUNE_MEMO_STATS["hit"] >= 1
    # a different shape is a different key
    n = len(pp_patch.AUTOTUNE_MEMO)
    tuner.choose_one("ppimpl_test_op", r2, cfg, [torch.zeros(32, 8)])
    assert len(pp_patch.AUTOTUNE_MEMO) == n + 1
