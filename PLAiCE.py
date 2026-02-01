import time
import argparse

from Canvas import Canvas
from Synchronizer import Synchronizer


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
    max_seconds = 5 *60
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
