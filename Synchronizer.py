import Canvas
import threading
import math
from typing import List, Dict

class Synchronizer:

    def __init__(self, canvas: Canvas, numAgents: int):
        self.canvas = canvas
        self.proposals = [] # queue of proposals from the agents
        self.proposal_lock = threading.Lock()
        self.threads = [] # proposal_threads
        self.agents = [] # agent objects
        self.numAgents = numAgents
        self.run_thread = None
        self.running = False
        self.agent_bounds = {}

    def initialize_agents(self):
        from agents.agent import Agent
        from agents.agent_state import AgentState
        from agents.model_interface import AgentModel
        import random

        self.agents = []
        self.threads = []

        for i in range(self.numAgents):
            state = AgentState(
                agent_id=i,
                fov_radius=3,
                temperature=0.5,
                bias_contrast=random.uniform(0.0, 1.0),
                bias_smoothness=random.uniform(0.0, 1.0),
                bias_edge=random.uniform(0.0, 1.0),
            )
            model = AgentModel()
            self.agents.append(Agent(state, model))
            self.threads.append(None)



    def propose(self, changes):
        # with self.lock:
            self.proposals.extend(changes)

    def _compute_slice_bounds(self, index, cols, rows, overlap_ratio=0.6):
        height = len(self.canvas.pixels)
        width = len(self.canvas.pixels[0]) if height > 0 else 0
        if width == 0 or height == 0:
            return (0, 0, 0, 0)

        col = index % cols
        row = index // cols

        slice_w = max(1, math.ceil(width / cols))
        slice_h = max(1, math.ceil(height / rows))

        x0 = col * slice_w
        y0 = row * slice_h
        x1 = min(width, x0 + slice_w)
        y1 = min(height, y0 + slice_h)

        overlap_w = int(slice_w * overlap_ratio)
        overlap_h = int(slice_h * overlap_ratio)

        x0 = max(0, x0 - overlap_w)
        y0 = max(0, y0 - overlap_h)
        x1 = min(width, x1 + overlap_w)
        y1 = min(height, y1 + overlap_h)

        return (x0, x1, y0, y1)

    def worker(self, agent, bounds):
        import numpy as np
        import random
        import time

        x0, x1, y0, y1 = bounds

        while True:
            height = len(self.canvas.pixels)
            width = len(self.canvas.pixels[0]) if height > 0 else 0
            if width == 0 or height == 0:
                time.sleep(0.01)
                continue
            if x1 <= x0 or y1 <= y0:
                time.sleep(0.01)
                continue

            x = random.randrange(x0, x1)
            y = random.randrange(y0, y1)

            canvas_snapshot = np.array(self.canvas.pixels, dtype=np.uint8)
            canvas_version = self.canvas.age

            proposal = agent.step(canvas_snapshot, (x, y), canvas_version)
            with self.proposal_lock:
                self.proposals.append(proposal)

            time.sleep(0.01)

    def start(self):
        if self.numAgents <= 0:
            return

        cols = int(math.sqrt(self.numAgents))
        if cols * cols < self.numAgents:
            cols += 1
        rows = math.ceil(self.numAgents / cols)

        for i in range(self.numAgents):
            bounds = self._compute_slice_bounds(i, cols, rows, overlap_ratio=0.6)
            self.agent_bounds[i] = bounds
            self.threads[i] = threading.Thread(
                target=self.worker,
                args=(self.agents[i], bounds),
            )

    def run(self):
        if self.running:
            return
        self.running = True

        # start spinning agents
        for thread in self.threads:
            thread.start()

        while self.running:
            with self.proposal_lock:
                batch = self.proposals.copy()
                proposals = []
            # Dict[Pos, Dict[rgb, int]]
            modified_pixels = dict()

            cur_age = self.canvas.age

            for p in batch:
                weight: int = p.canvas_version
                modified_pixels[p.region_id][p.rgb] = weight
            for pos, m in modified_pixels.items():
                x, y = pos
                tempR, tempG, tempB = 0, 0, 0
                sumWeights = 0
                for rgb, weight in m.items():
                    r, g, b = rgb
                    tempR += r * (weight / cur_age)
                    tempG += g * (weight / cur_age)
                    tempB += b * (weight / cur_age)

                    sumWeights += (weight / cur_age)
                resultR = tempR / sumWeights
                resultG = tempG / sumWeights
                resultB = tempB / sumWeights

                self.canvas.write(x, y, (resultR, resultG, resultB))

        # stop spinning agents
        for thread in self.threads:
            thread.join()

    def start_run(self):
        if self.run_thread is not None and self.run_thread.is_alive():
            return
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()

    def stop_run(self):
        self.running = False
        if self.run_thread is not None:
            self.run_thread.join()

    def read(self, agent_id, startX, startY, width, height):
        bounds = self.agent_bounds.get(agent_id)
        if bounds is None:
            return []

        x0, x1, y0, y1 = bounds

        req_x0 = max(x0, startX)
        req_y0 = max(y0, startY)
        req_x1 = min(x1, startX + width)
        req_y1 = min(y1, startY + height)

        if req_x1 <= req_x0 or req_y1 <= req_y0:
            return []

        return self.canvas.read(
            req_x0,
            req_y0,
            req_x1 - req_x0,
            req_y1 - req_y0,
        )





    #     while proposals_available:
    #         batch = get_proposals()
    #         for p in batch:
    #             if p.canvas_version_seen < current_version:
    #                 (p)
    #             resolve_conflicts(p)
    #         apply_updates()
    #         increment_canvas_version()
