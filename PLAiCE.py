import time
import argparse
import threading
import signal
import os

from Canvas import Canvas
from Synchronizer import Synchronizer


def _start_parent_watcher(sync: Synchronizer, interval: float = 1.0):
    """Start a background thread that watches the parent process and stops
    the synchronizer if the parent exits.

    Uses psutil if available for robust parent existence checks, otherwise
    falls back to polling os.getppid() (on Windows, getppid may return 0/1 when
    parent exits).
    """

    try:
        import psutil
    except Exception:
        psutil = None

    def _watcher():
        parent_pid = os.getppid()
        while sync.running:
            try:
                if psutil is not None:
                    try:
                        ps = psutil.Process(parent_pid)
                        if not ps.is_running() or ps.status() == psutil.STATUS_ZOMBIE:
                            print("[parent_watcher] parent process not running, stopping sync")
                            sync.stop_run()
                            break
                    except psutil.NoSuchProcess:
                        print("[parent_watcher] parent process disappeared, stopping sync")
                        sync.stop_run()
                        break
                else:
                    # Fallback: if parent pid becomes 1 (init) or 0, assume parent died.
                    cur_ppid = os.getppid()
                    if cur_ppid == 1 or cur_ppid == 0 or cur_ppid != parent_pid:
                        print(f"[parent_watcher] parent pid changed ({parent_pid} -> {cur_ppid}), stopping sync")
                        sync.stop_run()
                        break
            except Exception as exc:
                print(f"[parent_watcher] watcher exception: {exc}")
            time.sleep(interval)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


def _register_signal_handlers(sync: Synchronizer):
    def _handle(signum, frame):
        print(f"[signal] received signal {signum}, stopping sync")
        sync.stop_run()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            # Some platforms may not support setting signal handlers for all signals
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent pipeline logs")
    parser.add_argument("--preload", action="store_true", help="Preload models before starting workers")
    args = parser.parse_args()

    width = 256
    height = 256
    num_agents = 4

    canvas = Canvas(width, height)
    sync = Synchronizer(canvas, num_agents)
    sync.verbose = args.verbose
    sync.initialize_agents()
    if args.preload and sync.agents:
        # Warm up shared diffuser and classifier once to avoid per-thread load.
        agent0 = sync.agents[0]
        _ = agent0.diffuser._get_diffuser()
        _ = agent0.prompt_generator
    sync.start()
    sync.start_run()
    # Start parent watcher and signal handlers so child threads stop when
    # the parent process (this script) is terminated.
    _register_signal_handlers(sync)
    _start_parent_watcher(sync)
    max_seconds = 5 * 60
    start_time = time.time()
    last_log = start_time
    while sync.running and (time.time() - start_time) < max_seconds:
        now = time.time()
        if now - last_log >= 1.0:
            print(f"canvas age: {canvas.getAge()}")
            last_log = now
        time.sleep(0.05)
    if sync.running:
        sync.stop_run()
    else:
        sync.stop_run()
    canvas.export()


if __name__ == "__main__":
    main()
