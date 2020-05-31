import numpy as np
from matplotlib import pyplot as plt
import random
import json
import configparser
import os



def load_params( file_paths: dict ) -> dict:
    """

    input:
    file_paths: dictionary of all file paths in the module
    outputs:
    pars: dictionary of all paramters listed in params.ini
    """

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


def plot_model(     sr_log:    np.ndarray,
                    sr_mg_log: np.ndarray,
                    f:         np.ndarray,
                    f_avg:     np.ndarray,
                    name:      str,
                    pars:      dict):
    """

    Inputs:
    sr_log:     imported stimulated reps [time, squat, deadlift, pullup, bench]
    sr_mg_log:  stimulated reps for every muscle group
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    f:          total fatigue on muscle groups
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    f_avg:      moving average of total fatigue on muscle groups
                [time, quad, ham, abs, pec, bu, tri, lat, calf ]
    name:       name of plot
    pars:       all imported parameters

    """
    # stimulated reps in an exercise
    sr_t        = sr_log[:, 0]
    sr_squat    = sr_log[:, 1]
    sr_deadlift = sr_log[:, 2]
    sr_pullup   = sr_log[:, 3]
    sr_bench    = sr_log[:, 4]

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

    def init_pars(t, f_pars):
        """
        function to initalize mg parameters in correct format
        input:
            t:      array for time
            f_pars: dictonary for a specific muscle attribute.

        """
        ones = np.ones(t.shape)
        mat =np.array([
            t,
            ones*f_pars["quad"],
            ones*f_pars["ham"],
            ones*f_pars["abs"],
            ones*f_pars["pec"],
            ones*f_pars["bi"],
            ones*f_pars["tri"],
            ones*f_pars["lat"],
            ones*f_pars["calf"]])
        mat = mat.T
        return mat


    # desired fatigue
    f_ref         = init_pars(f[:,0], pars["f_d"])
    f_max         = init_pars(f[:,0], pars["f_max"])

    plt.rcParams['axes.labelsize'] = 10.0
    plt.rcParams['axes.titlesize'] = 11.0
    plt.rcParams['legend.fontsize'] = 10.0
    #plt.rcParams['figure.figsize'] = 25 / 2.54, 20 / 2.54
    plt.rcParams["figure.figsize"] = [16,9]


    def fatigue_subplot(t,
                        f_ex,
                        f_avg_ex,
                        f_ref_ex,
                        f_max_ex,
                        name: str,
                        subplot_id: int):


        plt.subplot(subplot_id)
        plt.plot(t, f_ex, label="fatigue" )
        plt.plot(t, f_ref_ex, label="desired fatigue" )
        plt.plot(t, f_max_ex, label="max fatigue" )
        plt.plot(t, f_avg_ex, label="avg fatigue" )
        plt.title(name)
        plt.grid()
        plt.legend()

    plt.figure(name)
    plt.clf()
    fatigue_subplot(f[:,0], f[:,1],   f_avg[:,1], f_ref[:,1],  f_max[:,1], "quad", 421)
    fatigue_subplot(f[:,0], f[:,2],   f_avg[:,2], f_ref[:,2],  f_max[:,2],  "ham",  422)
    fatigue_subplot(f[:,0], f[:,3],   f_avg[:,3], f_ref[:,3],  f_max[:,3],"abdom",423)
    fatigue_subplot(f[:,0], f[:,4],   f_avg[:,4], f_ref[:,4],  f_max[:,4],  "pec",  424)
    fatigue_subplot(f[:,0], f[:,5],   f_avg[:,5], f_ref[:,5],  f_max[:,5],   "bi",   425)
    fatigue_subplot(f[:,0], f[:,6],   f_avg[:,6], f_ref[:,6],  f_max[:,6],  "tri",  426)
    fatigue_subplot(f[:,0], f[:,7],   f_avg[:,7], f_ref[:,7],  f_max[:,7],  "lat",  427)
    fatigue_subplot(f[:,0], f[:,8],   f_avg[:,8], f_ref[:,8],  f_max[:,8], "calf", 428)

    plt.figure("squat_" + name)
    plt.plot(sr_t, sr_squat, label="sr_squat" )
    plt.scatter(sr_mg_t, sr_mg_quad, label="sr_quads")
    plt.xlabel("time [day]")
    plt.title("squat " + name)
    plt.legend()


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

def align_fatigue_pars(pars: dict):
    # desired fatigue
    f_d_pars = pars["f_d"]
    f_d =   np.array([
            f_d_pars["quad"],
            f_d_pars["ham"],
            f_d_pars["abs"],
            f_d_pars["pec"],
            f_d_pars["bi"],
            f_d_pars["tri"],
            f_d_pars["lat"],
            f_d_pars["calf"]])

    # desired fatigue
    f_max_pars = pars["f_max"]
    f_max = np.array([
            f_max_pars["quad"],
            f_max_pars["ham"],
            f_max_pars["abs"],
            f_max_pars["pec"],
            f_max_pars["bi"],
            f_max_pars["tri"],
            f_max_pars["lat"],
            f_max_pars["calf"]])
    return f_d, f_max

def compute_sr_d(f0: np.ndarray,
                 pars: dict):

    f_d, f_max = align_fatigue_pars(pars = pars)

    wo_opt = pars["wo_opt"]
    sr_max = 30
    N_ex = 4

    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])
    f_d = f_d[0]

    sr_log = None

    f_cost_values = [np.inf]
    for i in range(1000):
        # generate sr_log
        sr_log_cand = np.random.randint(0 , sr_max, (N_t,  N_ex + 1 ))
        sr_log_cand[:,0] = np.linspace(0, wo_opt["t_horizon"], N_t)

        sr_mg_log_cand, f_cand, f_avg_cand = compute_model(sr_log=sr_log_cand,
                                                           f0=None,
                                                           pars=pars)

        f_cost_cand = np.linalg.norm(f_avg_cand[:,1:] - f_d)

        if f_cost_cand < f_cost_values[-1]:
            sr_log = sr_log_cand
            f_cost_values.append(f_cost_cand)
            print("cost ", f_cost_values)


    return sr_log



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


if __name__=="__main__":
    # file paths
    file_paths = {}
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
    file_paths["params"] = os.path.join(file_paths["module"], "params.ini")
    file_paths["sr_log"] = os.path.join(file_paths["module"],"sr_log.csv")

    # parameters
    pars = load_params(file_paths=file_paths)

    # load stimulated reps log for exercises
    sr_log = import_log(file_paths)

    sr_mg_log, f, f_avg = compute_model(sr_log=sr_log,
                                        f0=None,
                                        pars=pars)

    sr_d_log = compute_sr_d(f0=f[:, -1],
                             pars=pars)

    sr_d_mg_log, f_d, f_d_avg = compute_model(sr_log=sr_d_log,
                                              f0=None,
                                              pars=pars)

    # plotting
    plot_model(sr_log, sr_mg_log, f, f_avg, "data", pars)
    plot_model(sr_d_log, sr_d_mg_log, f_d, f_d_avg, "desired", pars)

    plt.show()
