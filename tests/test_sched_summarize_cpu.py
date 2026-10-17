"""`bench.sched summarize` against a handoff that covers only some steps (v2-realistic: good only): the runs of the
other steps are listed as not compared instead of failing on the missing target, and the Notes part is kept."""
import json

from bench import sched


def test_summarize_skips_steps_without_targets(tmp_path, monkeypatch):
    handoff = tmp_path / "handoff_realistic"
    (handoff / "data").mkdir(parents=True)
    (handoff / "data" / "sched_check_targets.json").write_text(json.dumps({"good": {}}))
    (handoff / "data" / "force_check_targets.json").write_text(json.dumps({"good": {}}))
    (handoff / "data" / "session_targets.json").write_text(json.dumps([{"setting": "good_hbm", "max_sessions": 400}]))
    monkeypatch.setattr(sched, "HANDOFF", handoff)
    root = tmp_path / "runs"
    for name in ("sched_check_plain_7708", "force_check_even_7738", "sched_srpt8k_n500_7727"):
        (root / name).mkdir(parents=True)
        (root / name / "run.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    (out / "scheduler.md").write_text("old tables\n\n## Notes\n\nkept\n")
    text = sched.summarize(root, out).read_text()
    assert "handoff_realistic" in text and "not compared" in text
    for name in ("sched_check_plain_7708", "force_check_even_7738", "sched_srpt8k_n500_7727"):
        assert name in text
    assert "old tables" not in text and text.rstrip().endswith("kept")
