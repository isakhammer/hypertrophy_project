import numpy as np
from src import model

class Particle:
    b = None
    p = None
    p = None


def particle_swarm_optimization(f0: np.ndarray,
                                f_d: np.ndarray,
                                f_max: np.ndarray,
                                pars: dict):

    # options
    n = 1000
    sr_max = 30
    N_ex = 4

    # general options
    wo_opt = pars["wo_opt"]
    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])

    # generate inital value
    sr_log_cand = np.random.randint(0 , sr_max, (N_t,  N_ex + 1 ))
    sr_log_cand[:,0] = np.linspace(0, wo_opt["t_horizon"], N_t)

    sr_log = None

    f_cost_values = [np.inf]
    for i in range(n):
        # optimize here

    return sr_log
