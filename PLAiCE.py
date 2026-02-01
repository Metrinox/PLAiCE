import time

from Canvas import Canvas
from Synchronizer import Synchronizer


def main():
    # TODO: replace with real config / CLI flags
    width = 224
    height = 224
    num_agents = 16

    canvas = Canvas(width, height)
    sync = Synchronizer(canvas, num_agents)
    sync.initialize_agents()
    sync.start()
    sync.start_run()

    # TODO: replace with a real run condition or UI loop
    time.sleep(2.0)
    sync.stop_run()
    canvas.export()


if __name__ == "__main__":
    main()
