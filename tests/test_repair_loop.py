import threading

import pytest

from ts import repair_loop


def test_gpu_slot_hands_out_each_gpu_once(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_loop, "GPU_LOCK", tmp_path / "gpu{}.lock")
    with repair_loop.gpu_slot() as a, repair_loop.gpu_slot() as b:
        assert {a.serve_gpu, b.serve_gpu} == {0, 1} and a.port == 8000 + a.serve_gpu
        got = []
        waiter = threading.Thread(target=lambda: got.append(repair_loop.gpu_slot(wait_s=0.05).__enter__()))
        waiter.start()
        waiter.join(0.3)
        assert not got  # both GPUs taken: the third step waits
    waiter.join(2)
    assert got and got[0].serve_gpu in (0, 1)


def test_in_parallel_returns_results_and_raises_errors():
    assert repair_loop.in_parallel(lambda: 1, lambda: 2) == [1, 2]
    with pytest.raises(ValueError):
        repair_loop.in_parallel(lambda: 1, lambda: (_ for _ in ()).throw(ValueError("x")))


def test_round1_must_exist(tmp_path):
    with pytest.raises(SystemExit):
        repair_loop.round1_adapters(tmp_path, 1)
