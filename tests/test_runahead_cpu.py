"""RunAhead (ppimpl/runtime/worker.py): which earlier microbatches the main thread waits for before a launch."""

from ppimpl.runtime.worker import RunAhead


class Ev:
    def __init__(self, name, log, done=False):
        self.name, self.log, self.done = name, log, done

    def query(self):
        return self.done

    def synchronize(self):
        self.log.append(self.name)
        self.done = True


def launch(ra, log, kinds):
    """Launch microbatches of `kinds` ("g" graph, "e" eager) in order; returns what each launch waited for."""
    waited = []
    for k, kind in enumerate(kinds):
        n = len(log)
        ra.before(kind == "e")
        waited.append(log[n:])
        ra.after(Ev(k, log), kind == "e")
    return waited


def test_graph_microbatches_wait_for_k_minus_2():
    log = []
    assert launch(RunAhead(True), log, "gggg") == [[], [], [0], [1]]


def test_eager_waits_until_no_other_eager_is_queued():
    log = []
    # 1 (eager) after 0 (graph): nothing; 2 (eager): 0 (depth) and eager 1; 3 (graph): 1 is done already
    assert launch(RunAhead(True), log, "geeg") == [[], [], [0, 1], []]


def test_done_microbatches_are_not_waited_for():
    log = []
    ra = RunAhead(True)
    ra.after(Ev("a", log, done=True), True)
    ra.after(Ev("b", log, done=True), False)
    ra.before(True)
    assert log == [] and ra.waits == 0 and [e.name for e, _ in ra.q] == ["b"]


def test_cpu_never_waits():
    ra = RunAhead(False)
    ra.after(object(), True)
    ra.before(True)
    assert not ra.q
