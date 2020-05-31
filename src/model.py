import numpy as np

def compute_fatigue(sr_mg_log: np.ndarray,
                    f0: np.ndarray,
                    pars: dict ) -> np.ndarray:

    """
    input:
    f0:         initial start fatigue at sr_mg_log for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    sr_mg_log:  stimulated reps for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    pars:       dict of all parameters.

    output:
    fatigue:  total fatigue on muscle groups
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]

    """
    rec_rates = pars["rec_rates"]
    rec_rates = np.array([
                rec_rates["quad"],
                rec_rates["ham"],
                rec_rates["pec"],
                rec_rates["abs"],
                rec_rates["bi"] ,
                rec_rates["tri"],
                rec_rates["lat"],
                rec_rates["calf"]
                ])


    # time interval
    N = 500
    t_start = 0
    t_end = np.amax(sr_mg_log[:,0])
    t_interval = np.linspace(t_start, t_end, N)

    # number of muscle groups
    n_mg = sr_mg_log.shape[1] - 1

    f = np.zeros((N, sr_mg_log.shape[1]))
    f[:,0] = t_interval

    # initalize time and fatigue at workout j
    j_sr = 0
    f_j_sr = np.zeros(sr_mg_log.shape[1])
    t_j_sr = 0

    # Set inital start fatigue
    if f0 is not None:
        f_j_sr[0,:] = f0


    for i in range(N-1):
        t_i = t_interval[i]

        # saving fatigue at previous workout
        if j_sr < sr_mg_log.shape[0] and sr_mg_log[j_sr, 0] < t_i:
            f[i, 1:] += sr_mg_log[j_sr, 1:]
            f_j_sr= f[i, :]
            t_j_sr =  f_j_sr[0]
            j_sr += 1
        f[i+1, 1:] = f_j_sr[1:]*np.exp(-(t_i - t_j_sr)*rec_rates)

    return f

def compute_sr_mg_log(sr_log: np.ndarray,
                      pars: dict):

    """
    Compute stimulated reps for each muscle group given stimulated reps from the exercises.

    input:
    pars: dict of all parameters
    sr_log:   imported stimulated reps [time, squat, deadlift, pullup, bench]

    """

    def init_opt(ex_opt):
        opt = [
            ex_opt["quad"],
            ex_opt["ham"],
            ex_opt["abs"],
            ex_opt["pec"],
            ex_opt["bi"],
            ex_opt["tri"],
            ex_opt["lat"],
            ex_opt["calf"]
        ]
        return opt

    # Arrays of how much each exercise if affecting muscle groups
    ex_opts = pars["ex"]
    squat    = init_opt(ex_opts["squat"])
    deadlift = init_opt(ex_opts["deadlift"])
    bench    = init_opt(ex_opts["bench"])
    pullup   = init_opt(ex_opts["pullup"])

    # initalize transformation matrix
    T_mg_ex = np.array([
                    squat,
                    deadlift,
                    bench,
                    pullup
                        ])

    # extract time and exercises
    t_sr = sr_log[:, 0]
    sr_ex = sr_log[:,1:]
    sr_mg = np.dot(sr_ex, T_mg_ex)

    # Adds time to log for muscle groups
    sr_mg_log = np.zeros((sr_mg.shape[0], sr_mg.shape[1] + 1))
    sr_mg_log[:, 0] = t_sr
    sr_mg_log[:, 1:] = sr_mg

    return sr_mg_log

def compute_fatigue_avg(f:      np.ndarray,
                        pars:   dict) -> np.ndarray:

    t_moving_avg = pars["wo_opt"]["t_moving_avg"]
    f_avg = np.zeros(f.shape)
    f_avg[:,0] = f[:,0]

    for i in range(1, f.shape[0]):
        t1 = f[i,0]
        t0 = max(f[0,0], t1 - t_moving_avg)
        k = (np.abs(f[:i,0] - t0)).argmin()
        f_avg[i,1:] = np.mean(f[k:i,1:], axis=0)

    return f_avg

def compute_model(sr_log: np.ndarray,
                 f0: np.ndarray,
                 pars: dict):

    # transform stimulated reps from ecercise to muscle group
    sr_mg_log = compute_sr_mg_log(sr_log=sr_log,
                                  pars=pars)

    # compute muscle fatigue
    f = compute_fatigue(sr_mg_log=sr_mg_log,
                        f0=f0,
                        pars=pars)

    # computing moving average of total muscle fatigue
    f_avg = compute_fatigue_avg(f=f,
                                pars=pars)

    return sr_mg_log, f, f_avg

