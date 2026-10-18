"""`kvpool.writeback.WriteBack` on a toy level (lanes/m6 TIERS.md §5): HBM's headroom rule (C - F - D >= R, R the
largest demand in a window of 2T, T the longest write) and DRAM's write at eviction."""

from ppimpl.kvpool.writeback import WriteBack


class Toy:
    """A level of `capacity` charge: entries in LRU order (dict order), `fix` charge held outside them. A write
    returns a token unless the key is in `below` (held there whole already)."""

    def __init__(self, capacity, entries=()):
        self.capacity, self.fix, self.entries, self.below, self.written = capacity, 0, dict(entries), set(), []

    def fixed(self):
        return self.fix

    def lru(self):
        return iter(self.entries)

    def write(self, key, entry=None):
        self.written.append(key)
        return None if key in self.below else object()


class Clock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns


def hbm(capacity, sizes):
    lvl, clock = Toy(capacity, sizes), Clock()
    wb = WriteBack("hbm", lvl, ahead=True, clock=clock)
    for k, c in sizes.items():
        wb.add(k, c)
    return lvl, wb, clock


def land(wb, key, clock, ns):
    clock.ns = ns
    wb.done(key, wb.writing[key][0])


def test_headroom_is_the_largest_demand_and_the_lru_tail_is_written_until_it_is_free_or_clean():
    lvl, wb, clock = hbm(100, {1: 10, 2: 10, 3: 10, 4: 10})
    wb.step()                              # demand 40 (F + D grew from 0): R = 40; C - F - D = 60 >= 40
    assert wb.headroom == 40 and lvl.written == []
    clock.ns = 10
    lvl.fix = 30                           # admissions of 30
    wb.step()                              # T = 0: the window holds this step only; R stays 40 (running max)
    assert wb.headroom == 40               # 100 - 30 - 40 = 30 < 40: write 1 (LRU) -> 40
    assert lvl.written == [1] and wb.stats["ahead_writes"] == 1 and not wb.evictable(1)
    land(wb, 1, clock, 30)                 # T = 20 ns
    assert wb.write_ns == 20 and wb.evictable(1)


def test_demand_within_2T_adds_up_and_older_demand_leaves_the_window():
    lvl, wb, clock = hbm(1000, {})
    lvl.entries[0] = 1
    wb.add(0, 1)
    wb.step()
    clock.ns = 1
    wb._write_lru(1, "ahead_writes")
    land(wb, 0, clock, 101)                 # T = 100: window 200
    for t, f in ((200, 50), (300, 100), (390, 130)):
        clock.ns, lvl.fix = t, f
        wb.step()
    assert wb.headroom == 130               # 50 + 50 + 30 within [190, 390]; the landing is not counted as supply
    clock.ns, lvl.fix = 1000, 140
    wb.step()                              # alone in its window: the running max stays
    assert wb.headroom == 130


def test_a_write_that_moves_nothing_counts_as_landed_not_as_negative_demand():
    lvl, wb, clock = hbm(100, {1: 30, 2: 30})
    lvl.below.add(1)                       # 1 is held below already
    lvl.fix = 20
    wb.step()                              # demand 80: R = 80, 100 - 20 - 60 < 80: write 1 (clean at once), 2
    assert lvl.written == [1, 2] and wb.evictable(1) and not wb.evictable(2)
    clock.ns = 5
    lvl.fix = 25
    wb.step()                              # demand 5 (not 5 - 30)
    assert wb._demand[-1] == (5, 5)


def test_make_room_writes_beyond_the_writes_in_flight():
    lvl, wb, clock = hbm(100, {1: 10, 2: 10, 3: 10})
    wb._write_lru(10, "ahead_writes")      # 1 in flight
    wb.make_room(25)                       # 25 short, 10 in flight: 2 and 3 start
    assert lvl.written == [1, 2, 3] and wb.stats["evict_writes"] == 2 and wb.stats["evict_waits"] == 1
    wb.make_room(25)                       # all in flight: nothing more
    assert wb.stats["evict_writes"] == 2


def test_at_eviction_only_level_never_writes_ahead():
    lvl, clock = Toy(10, {1: 5, 2: 5}), Clock()
    wb = WriteBack("dram", lvl, ahead=False, clock=clock)
    wb.add(1, 5)
    wb.add(2, 5)
    wb.step()
    assert lvl.written == [] and wb.headroom == 0
    assert wb.evicting(1, None) == "dirty" and wb.stats == dict(ahead_writes=0, evict_writes=1, evict_waits=1)
    lvl.below.add(2)                       # the tier below holds 2 whole: nothing moves, no wait
    assert wb.evicting(2, None) == "clean" and wb.stats["evict_waits"] == 1
