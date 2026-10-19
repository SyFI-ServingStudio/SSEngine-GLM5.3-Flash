from ssengine.runtime.transport import (Chunk, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_controller,
                                      decode_to_worker, encode, for_stage)


def test_round_trip():
    plan = Plan(7, [Chunk("a", 0, 3, [1], [2, 3, 4, 5, 6, 7], True, [10, 11, 12]),
                    Chunk("b", 8192, 100, [8, 9], [10, 11, 12, 13, 14, 15], False, list(range(100)))])
    assert decode_to_worker(encode(plan)) == plan
    assert plan.num_tokens == 103
    assert isinstance(decode_to_worker(encode(Shutdown())), Shutdown)
    for msg in (Ready(2, 7144), StageDone(3, 7, 10, 20, [("a", 5)]), StageDone(1, 7, 10, 20, None),
                StageDone(0, 7, 10, 20, None, (1, 2, 3, 4, 5)),
                WorkerError(0, "boom")):
        assert decode_to_controller(encode(msg)) == msg


def test_token_ids_go_to_stage0_only():
    plan = Plan(0, [Chunk("a", 0, 3, [1], [2] * 6, True, [10, 11, 12])])
    assert for_stage(plan, 0).chunks[0].token_ids == [10, 11, 12]
    assert for_stage(plan, 2).chunks[0].token_ids is None
    assert plan.chunks[0].token_ids == [10, 11, 12]
