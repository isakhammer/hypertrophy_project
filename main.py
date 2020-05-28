import numpy as np
from matplotlib import pyplot as plt
import random
import json
import configparser
import os



def load_params( file_paths ):

    # load vehicle parameter file into a "pars" dict
    parser = configparser.ConfigParser()

    if not parser.read(file_paths["params"]):
        raise ValueError('Specified config file does not exist or is empty!')

    pars = {}
    pars["wo_opt"]     = json.loads(parser.get('SR_OPTIONS', 'wo_opt'))
    pars["f_max"]       = json.loads(parser.get('FATIGUE_OPTIONS', 'f_max'))
    pars["f_d"]         = json.loads(parser.get('FATIGUE_OPTIONS', 'f_d'))
    pars["rec_rates"]   = json.loads(parser.get('FATIGUE_OPTIONS', 'rec_rates'))

    ex = {}
    ex['squat']     = json.loads(parser.get('EXERCISE_OPTIONS', 'squat'))
    ex['deadlift']  = json.loads(parser.get('EXERCISE_OPTIONS', 'deadlift'))
    ex['bench']     = json.loads(parser.get('EXERCISE_OPTIONS', 'bench'))
    ex['pullup']    = json.loads(parser.get('EXERCISE_OPTIONS', 'pullup'))

    pars["ex"] = ex
    return pars

def import_log( file_paths: dict) -> np.ndarray:
    """
    Imports file path of raw data with csv-format containing:
    #time;squat;deadlift;pullup;bench

    Outputs:
    sr_log:   imported stimulated reps [time, squat, deadlift, pullup, bench]
    """
    def skipper(fname, header=False):
        """
        Function for skipping header

        """
        with open(fname) as fin:
            #no_comments = (line for line in fin if not line.lstrip().startswith('#'))
            if header:
                next(fin, None) # skip header
                for row in fin:
                    yield row



    # load data from csv file
    csv_data_tmp = np.loadtxt(skipper(file_paths["sr_log"], header=True), delimiter=';')

    # get sr log out of array.
    if np.shape(csv_data_tmp)[1] == 5:
        time         = csv_data_tmp[:,0]
        squat       = csv_data_tmp[:,1]
        deadlift    = csv_data_tmp[:,2]
        pullup      = csv_data_tmp[:,3]
        bench       = csv_data_tmp[:,4]

    else:
        raise IOError(file_paths["sr_log"] + " cannot be read!")

    # assemble to a single array
    sr_log = np.column_stack((time,
                                squat,
                                deadlift,
                                pullup,
                                bench
                                ))

    return sr_log


def compute_fatigue( sr_mg_log: np.ndarray, pars: dict ) -> np.ndarray:

    """
    input:
    pars:       dict of all parameters.
    sr_mg_log:  stimulated reps for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
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
    N = 1000
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



def compute_sr_mg_log(sr_log, pars):

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


def plot_fatigue(   sr_log:    np.ndarray,
                    sr_mg_log: np.ndarray,
                    f:         np.ndarray,
                    pars:      dict):
    """

    Inputs:
    sr_log:   imported stimulated reps [time, squat, deadlift, pullup, bench]
    sr_mg_log:  stimulated reps for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    fatigue:  total fatigue on muscle groups
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]

    """

    # fatigue
    f_t       = f[:, 0]
    f_quad    = f[:, 1]
    f_ham     = f[:, 2]
    f_abdom   = f[:, 3]
    f_pec     = f[:, 4]
    f_bi      = f[:, 5]
    f_tri     = f[:, 6]
    f_lat     = f[:, 7]
    f_calf    = f[:, 8]

    # helper array
    ones = np.ones(f_t.shape)

    # desired fatigue
    f_d         = pars["f_d"]
    f_d_quad    = ones*f_d["quad"]
    f_d_ham     = ones*f_d["ham"]
    f_d_abdom   = ones*f_d["abs"]
    f_d_pec     = ones*f_d["pec"]
    f_d_bi      = ones*f_d["bi"]
    f_d_tri     = ones*f_d["tri"]
    f_d_lat     = ones*f_d["lat"]
    f_d_calf    = ones*f_d["calf"]

    # max fatigue
    f_max         = pars["f_max"]
    f_max_quad    = ones*f_max["quad"]
    f_max_ham     = ones*f_max["ham"]
    f_max_abdom   = ones*f_max["abs"]
    f_max_pec     = ones*f_max["pec"]
    f_max_bi      = ones*f_max["bi"]
    f_max_tri     = ones*f_max["tri"]
    f_max_lat     = ones*f_max["lat"]
    f_max_calf    = ones*f_max["calf"]

    # stimulated reps for muscle groups
    sr_mg_t       = sr_mg_log[:, 0]
    sr_mg_quad    = sr_mg_log[:, 1]
    sr_mg_ham     = sr_mg_log[:, 2]
    sr_mg_abdom   = sr_mg_log[:, 3]
    sr_mg_pec     = sr_mg_log[:, 4]
    sr_mg_bi      = sr_mg_log[:, 5]
    sr_mg_tri     = sr_mg_log[:, 6]
    sr_mg_lat     = sr_mg_log[:, 7]
    sr_mg_calf    = sr_mg_log[:, 8]


    plt.figure("quads")
    plt.plot(f_t, f_quad, label="fatigue" )
    plt.plot(f_t, f_d_quad, label="desired fatigue" )
    plt.plot(f_t, f_max_quad, label="max fatigue" )
    plt.scatter(sr_mg_t, sr_mg_quad, label="simulated reps")
    plt.xlabel("time [day]")
    plt.title("quads")
    plt.legend()

def plot_sr(        sr_log:    np.ndarray,
                    sr_mg_log: np.ndarray):
    """

    Inputs:
    sr_log:   imported stimulated reps [time, squat, deadlift, pullup, bench]
    sr_mg_log:  stimulated reps for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]

    """
    sr_t        = sr_mg_log[:, 0]
    sr_squat    = sr_log[:, 1]
    sr_deadlift = sr_log[:, 2]
    sr_pullup   = sr_log[:, 3]
    sr_bench    = sr_log[:, 4]

    sr_mg_t       = sr_mg_log[:, 0]
    sr_mg_quad    = sr_mg_log[:, 1]
    sr_mg_ham     = sr_mg_log[:, 2]
    sr_mg_abdom   = sr_mg_log[:, 3]
    sr_mg_pec     = sr_mg_log[:, 4]
    sr_mg_bi      = sr_mg_log[:, 5]
    sr_mg_tri     = sr_mg_log[:, 6]
    sr_mg_lat     = sr_mg_log[:, 7]
    sr_mg_calf    = sr_mg_log[:, 8]


    plt.figure("comparison")
    plt.plot(sr_t, sr_squat, label="sr_squat" )
    plt.scatter(sr_mg_t, sr_mg_quad, label="sr_quads")
    plt.xlabel("time [day]")
    plt.title("comparison")
    plt.legend()


if __name__=="__main__":
    # file paths
    file_paths = {}
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
    file_paths["params"] = os.path.join(file_paths["module"], "params.ini")
    file_paths["sr_log"] = os.path.join(file_paths["module"],"sr_log.csv")

    # parameters
    pars = load_params(file_paths)

    # load stimulated reps log for exercises
    sr_log = import_log(file_paths)

    # transform stimulated reps from ecercise to muscle group
    sr_mg_log = compute_sr_mg_log(sr_log, pars)

    # compute muscle fatigue
    f = compute_fatigue(sr_mg_log, pars)

    # plotting
    plot_fatigue(sr_log, sr_mg_log, f, pars)
    plot_sr(sr_log, sr_mg_log)
    plt.show()















