import Canvas
import threading
import math
import time
import os
from typing import List, Dict

class Synchronizer:

    def __init__(self, canvas: Canvas, numAgents: int):
        self.canvas = canvas
        self.proposals = [] # queue of proposals from the agents
        self.proposal_lock = threading.Lock()
        self.proposal_cv = threading.Condition(self.proposal_lock)
        self.threads = [] # proposal_threads
        self.agents = [] # agent objects
        self.numAgents = numAgents
        self.run_thread = None
        self.running = False
        self.agent_bounds = {}
        self.verbose = False
        self.batch_index = 0

    def initialize_agents(self):
        from agents.agent import Agent
        from agents.agent_state import AgentState
        from agents.model_interface import AgentModel
        from agents.pipeline import PipelineConfig
        import random

        self.agents = []
        self.threads = []

        for i in range(self.numAgents):
            state = AgentState(
                agent_id=i,
                temperature=0.5,
                bias_contrast=random.uniform(0.0, 1.0),
                bias_smoothness=random.uniform(0.0, 1.0),
                bias_edge=random.uniform(0.0, 1.0),
                verbose=self.verbose,
            )
            model = AgentModel()
            pipeline_config = PipelineConfig(image_size=64)
            self.agents.append(Agent(state, model, pipeline_config=pipeline_config))
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
        import random
        import time

        x0, x1, y0, y1 = bounds

        while self.running:
            try:
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

                canvas_version = self.canvas.age

                fov = self.canvas.read(x0, y0, x1 - x0, y1 - y0)
                proposals = agent.step(fov, (x0, y0), canvas_version)
                if proposals:
                    with self.proposal_cv:
                        self.proposals.extend(proposals)
                        self.proposal_cv.notify()
                else:
                    if self.verbose:
                        now = time.time()
                        last = getattr(agent, "_last_empty_log", 0.0)
                        if now - last >= 1.0:
                            print(
                                f"[worker {agent.state.agent_id}] zero proposals; "
                                f"fov={len(fov)}x{len(fov[0]) if fov else 0}"
                            )
                            agent._last_empty_log = now

                time.sleep(0.01)
            except Exception as exc:
                print(f"[worker {agent.state.agent_id}] exception: {exc}")
                time.sleep(0.1)

    def start(self):
        if self.numAgents <= 0:
            return

        cols = int(math.sqrt(self.numAgents))
        if cols * cols < self.numAgents:
            cols += 1
        rows = math.ceil(self.numAgents / cols)

        for i in range(self.numAgents):
            bounds = self._compute_slice_bounds(i, cols, rows, overlap_ratio=0.4)
            self.agent_bounds[i] = bounds
            self.agents[i].state.slice_bounds = bounds
            self.threads[i] = threading.Thread(
                target=self.worker,
                args=(self.agents[i], bounds),
            )

    def run(self):
        print("[run] started")
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        # start spinning agents
        for thread in self.threads:
            thread.start()

        while self.running:
            if self.canvas.getAge() >= 512:
                print("[run] age limit reached, stopping")
                self.running = False
                break
            with self.proposal_cv:
                if not self.proposals:
                    self.proposal_cv.wait(timeout=2)

                batch = self.proposals.copy()
                self.proposals.clear()
            if batch:
                print(f"[run] batch size: {len(batch)}")
                if self.verbose:
                    sample = [ (p.region_id, p.rgb, p.canvas_version) for p in batch[:5] ]
                    print(f"[run] sample proposals (first 5): {sample}")
            # Dict[Pos, Dict[rgb, int]]
            modified_pixels = dict()

            cur_age = self.canvas.age

            for p in batch:
                # use proposal confidence as a weight; canvas_version can be 0
                # which would otherwise zero-out contributions and prevent updates
                weight = getattr(p, "confidence", None)
                if weight is None:
                    # fallback to 1.0 for older proposals
                    weight = 1.0
                # ensure non-zero small floor
                try:
                    weight = float(weight)
                except Exception:
                    weight = 1.0
                if weight <= 0.0:
                    weight = 0.01
                if p.region_id not in modified_pixels:
                    modified_pixels[p.region_id] = {}
                modified_pixels[p.region_id][p.rgb] = (
                    modified_pixels[p.region_id].get(p.rgb, 0) + weight
                )
            for pos, m in modified_pixels.items():
                x, y = pos
                tempR, tempG, tempB = 0, 0, 0
                sumWeights = 0
                for rgb, weight in m.items():
                    r, g, b = rgb
                    tempR += r * weight 
                    tempG += g * weight
                    tempB += b * weight 

                    sumWeights += weight
                if sumWeights == 0:
                    continue
                resultR = tempR / sumWeights
                resultG = tempG / sumWeights
                resultB = tempB / sumWeights

                # record previous value for debugging
                try:
                    prev = self.canvas.pixels[y][x]
                except Exception:
                    prev = None

                new_col = (int(resultR), int(resultG), int(resultB))
                self.canvas.write(x, y, new_col)

                # log a few sample modifications when verbose
                if self.verbose and (self.canvas.age % 10 == 0):
                    print(f"[run] modify pos={(x,y)} prev={prev} -> new={new_col}")
            if self.verbose:
                print(f"[run] modified_pixels count: {len(modified_pixels)}")
            self.canvas.increment_age()
            frame_path = os.path.join(frames_dir, f"frame_{self.canvas.age:04d}.png")
            self.canvas.export(frame_path)


        # stop spinning agents
        for thread in self.threads:
            thread.join()
        print("[run] stopped")

    def start_run(self):
        if self.run_thread is not None and self.run_thread.is_alive():
            return
        self.running = True
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
