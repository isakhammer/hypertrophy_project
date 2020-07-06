import numpy as np

def compute_fatigue(sr_mg_log: np.ndarray,
                    f0: np.ndarray,
                    pars: dict ) -> np.ndarray:

    """
    Input:
        sr_ex_log:  [t, sr_ex0, sr_ex1, ... ]       (N_t, N_ex + 1)
        f0:         [f0_mg0, f0_mg1, ... ]          (N_mg, )
        pars: dict of all parameters
    Out:
        f_log:      [t, f_mg0, f_mg1, ... ]         (N_t, N_mg + 1)

    """

    rec_rates = pars["rec_rates"]
    rec_tmp = []
    for mg_name in rec_rates:
        rec_tmp.append(rec_rates[mg_name])

    rec_rates = np.array(rec_tmp)


    # time interval
    N = 500
    t_start = 0
    t_end = np.amax(sr_mg_log[:,0])
    t_interval = np.linspace(t_start, t_end, N)

    # number of muscle groups
    n_mg = sr_mg_log.shape[1] - 1

    f_log = np.zeros((N, sr_mg_log.shape[1]))
    f_log[:,0] = t_interval

    # initalize time and fatigue at workout j
    j_sr = 0
    f_j_sr = np.zeros(sr_mg_log.shape[1])
    t_j_sr = 0

    # Set inital start fatigue
    if f0 is not None:
        f_j_sr[1:] = f0
        f_log[0, 1:] = f_j_sr[1:]

    for i in range(N-1):
        t_i = t_interval[i]

        # saving fatigue at previous workout
        if j_sr < sr_mg_log.shape[0] and sr_mg_log[j_sr, 0] < t_i:
            f_log[i, 1:] += sr_mg_log[j_sr, 1:]
            f_j_sr= f_log[i, :]
            t_j_sr =  f_j_sr[0]
            j_sr += 1
        f_log[i+1, 1:] = f_j_sr[1:]*np.exp(-(t_i - t_j_sr)*rec_rates)

    return f_log

def compute_sr_mg_log(sr_ex_log: np.ndarray,
                      pars: dict):

    """
    Compute stimulated reps for each muscle group given stimulated reps from the exercises.

    input:
    pars: dict of all parameters
    sr_log:   imported stimulated reps [time, squat, deadlift, pullup, bench]

    """


    # Arrays of how much each exercise if affecting muscle groups
    ex_pars = pars["ex"]

    # initalize transformation matrix
    T_tmp = []
    for ex_name in ex_pars:
        ex = []
        ex_par = ex_pars[ex_name]

        for mg_name in ex_par:
            mg_value = ex_par[mg_name]
            ex.append(mg_value)

        T_tmp.append(ex)
    T_mg_ex = np.array(T_tmp)

    # extract time and exercises
    t_sr = sr_ex_log[:, 0]
    sr_ex = sr_ex_log[:,1:]
    sr_mg = np.dot(sr_ex, T_mg_ex)

    # Adds time to log for muscle groups
    sr_mg_log = np.zeros((sr_mg.shape[0], sr_mg.shape[1] + 1))
    sr_mg_log[:, 0] = t_sr
    sr_mg_log[:, 1:] = sr_mg

    return sr_mg_log

def compute_fatigue_avg(f_log:  np.ndarray,
                        pars:   dict) -> np.ndarray:

    delta_t_avg = pars["wo_opt"]["delta_t_avg"]

    f_avg_log = np.zeros(f_log.shape)
    f_avg_log[:,0] = f_log[:,0]

    for i in range(1, f_log.shape[0]):

        t1 = f_log[i,0]
        t0 = max(f_log[0,0], t1 - delta_t_avg)

        # computes index k at t0
        k = (np.abs(f_log[:i,0] - t0)).argmin()

        # computes average fatigue from t0 to t1
        f_avg_log[i,1:] = np.mean(f_log[k:i,1:], axis=0)

    return f_avg_log

def compute_model(sr_ex_log: np.ndarray,
                 f0:        np.ndarray,
                 pars: dict):
    """
    Computing the fatigue, moving average fatigue and stimulated reps for each muscle group given a initial fatigue f0 at 0 and total of stimulated reps.

    Input:
        sr_ex_log:  [t, sr_ex0, sr_ex1, ... ]       (N_t, N_ex + 1)
        f0:         [f0_mg0, f0_mg1, ... ]          (N_mg, )
        pars: dict of all parameters
    Out:
        sr_mg_log:  [t, sr_mg0, sr_mg1, ... ]       (N_t, N_mg + 1)
        f_log:      [t, f_mg0, f_mg1, ... ]         (N_t, N_mg + 1)
        f_avg_log:  [t, f_mg0, f_mg1, ... ]         (N_t, N_mg + 1)

    """


    # transform stimulated reps from ecercise to muscle group
    sr_mg_log = compute_sr_mg_log(sr_ex_log=sr_ex_log,
                                  pars=pars)

    # compute muscle fatigue
    f_log = compute_fatigue(sr_mg_log=sr_mg_log,
                            f0=f0,
                        pars=pars)

    # computing moving average of total muscle fatigue
    f_avg_log = compute_fatigue_avg(f_log=f_log,
                                pars=pars)


    return sr_mg_log, f_log, f_avg_log

