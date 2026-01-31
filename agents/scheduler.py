import random

def select_agents(agents, fraction=0.3):
    """
    Randomly select a subset of agents to run.
    """
    k = max(1, int(len(agents) * fraction))
    return random.sample(agents, k)
