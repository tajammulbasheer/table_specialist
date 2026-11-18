"""The watcher must wait for a real process, and must not wait for itself."""
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watch_until_done.sh"


def watch(pattern, interval="0.1"):
    return subprocess.Popen(["bash", "-c", f'{SCRIPT} test "{pattern}" {interval}'], stdout=subprocess.PIPE, text=True)


def test_exits_at_once_when_nothing_matches():
    watcher = watch("no_such_process_abc123")
    assert watcher.wait(timeout=5) == 0
    assert "test finished" in watcher.stdout.read()


def test_waits_for_a_matching_process_then_exits():
    target = subprocess.Popen(["bash", "-c", "exec -a target_proc_xyz789 sleep 60"])
    try:
        time.sleep(0.2)
        watcher = watch("target_proc_xyz789")
        time.sleep(0.5)
        assert watcher.poll() is None  # still waiting
    finally:
        target.kill()
        target.wait()
    assert watcher.wait(timeout=5) == 0
