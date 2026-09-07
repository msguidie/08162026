"""Throughput guard for the training hot path."""

from splendor_ai.validation.replay_check import bench

#: docs/PLAN.md targets >= 30k apply+legal_mask per second per core; the
#: assertion is set lower so a busy CI box does not fail the build, but the
#: measured number is printed.
MIN_STEPS_PER_SECOND = 15000


def test_apply_and_legal_mask_throughput(capsys):
    result = bench(60000)
    with capsys.disabled():
        print(f"\n  apply+legal_mask: {result['per_second']:,.0f} steps/s "
              f"({result['calls']} calls in {result['seconds']:.2f}s)")
    assert result["per_second"] > MIN_STEPS_PER_SECOND


def test_clone_is_cheap():
    import random
    import time
    from splendor_ai.rules import engine as E
    s = E.new_game(4, "TEAM", "ADJACENT", rng=random.Random(1))
    for _ in range(60):
        acts = E.legal_actions(s)
        if not acts or s.is_over():
            break
        E.apply(s, acts[0])
    t0 = time.perf_counter()
    for _ in range(20000):
        s.clone()
    dt = time.perf_counter() - t0
    assert 20000 / dt > 20000, f"clone throughput {20000 / dt:,.0f}/s"
