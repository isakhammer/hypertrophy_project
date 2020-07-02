import numpy as np
from src import model


def brute_force_optimization(f0:    np.ndarray,
                             f_d:   np.ndarray,
                             f_max: np.ndarray,
                             pars: dict):
    """
    Input:
        f0_d:       [sr_ex0, sr_ex1, ... ]          (N_ex, )
        f_d:        [f_d_mg0, f_d_mg1, ... ]        (N_mg, )
        f_max:      [f_d_mg0, f_d_mg1, ... ]        (N_mg, )
        pars: dict of all parameters
    Out:
        sr_ex_log:  [t, sr_ex0, sr_ex1, ... ]       (N_t, N_ex + 1)
    """
    # options
    n = 1000
    sr_max = 30
    N_ex = len(pars["ex_names"])

    # general options
    wo_opt = pars["wo_opt"]
    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])

    sr_ex_log = None

    f_cost_values = [np.inf]
    for i in range(n):
        # generate sr_ex_log
        sr_ex_log_cand = np.random.randint(0 , sr_max, (N_t,  N_ex + 1 ))
        sr_ex_log_cand[:,0] = np.linspace(0, wo_opt["t_horizon"], N_t)

        sr_mg_log_cand, f_cand, f_avg_cand = model.compute_model(sr_ex_log=sr_ex_log_cand,
                                                                f0=f0,
                                                                pars=pars)

        f_cost_cand = np.linalg.norm(f_avg_cand[:,1:] - f_d)

        if f_cost_cand < f_cost_values[-1]:
            sr_ex_log = sr_ex_log_cand
            f_cost_values.append(f_cost_cand)
            print(i, f_cost_values[-1])

    return sr_ex_log
