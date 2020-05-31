import numpy as np
from src import model


def brute_force_optimization(f0: np.ndarray,
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

    sr_log = None

    f_cost_values = [np.inf]
    for i in range(n):
        # generate sr_log
        sr_log_cand = np.random.randint(0 , sr_max, (N_t,  N_ex + 1 ))
        sr_log_cand[:,0] = np.linspace(0, wo_opt["t_horizon"], N_t)

        sr_mg_log_cand, f_cand, f_avg_cand = model.compute_model(sr_log=sr_log_cand,
                                                                f0=None,
                                                                pars=pars)

        f_cost_cand = np.linalg.norm(f_avg_cand[:,1:] - f_d)

        if f_cost_cand < f_cost_values[-1]:
            sr_log = sr_log_cand
            f_cost_values.append(f_cost_cand)
            print(i, f_cost_values[-1])

    return sr_log
