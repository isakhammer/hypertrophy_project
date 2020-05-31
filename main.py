import numpy as np
from matplotlib import pyplot as plt
from src import model
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


    def fatigue_subplot(f_t,
                        f_mg,
                        f_avg_mg,
                        f_ref_mg,
                        f_max_mg,
                        sr_mg_t,
                        sr_mg,
                        name: str,
                        subplot_id: int):


        plt.subplot(subplot_id)
        plt.plot(f_t,       f_mg,       label="f" )
        plt.plot(f_t,       f_ref_mg,   label="f_ref" )
        plt.plot(f_t,       f_max_mg,   label="f_max" )
        plt.plot(f_t,       f_avg_mg,   label="f_avg" )
        plt.scatter(sr_mg_t,   sr_mg,   label="sr_mg", s=7, color="black")
        plt.ylim((0,25))
        plt.title(name)
        plt.grid()
        plt.legend()

    plt.figure(name)
    plt.clf()
    fatigue_subplot(f[:,0], f[:,1],   f_avg[:,1], f_ref[:,1],  f_max[:,1],  sr_mg_log[:,0],  sr_mg_log[:,1], "quad",    421)
    fatigue_subplot(f[:,0], f[:,2],   f_avg[:,2], f_ref[:,2],  f_max[:,2],  sr_mg_log[:,0],  sr_mg_log[:,2],  "ham",    422)
    fatigue_subplot(f[:,0], f[:,3],   f_avg[:,3], f_ref[:,3],  f_max[:,3],  sr_mg_log[:,0],  sr_mg_log[:,3],"abdom",    423)
    fatigue_subplot(f[:,0], f[:,4],   f_avg[:,4], f_ref[:,4],  f_max[:,4],  sr_mg_log[:,0],  sr_mg_log[:,4],  "pec",    424)
    fatigue_subplot(f[:,0], f[:,5],   f_avg[:,5], f_ref[:,5],  f_max[:,5],  sr_mg_log[:,0],  sr_mg_log[:,5],   "bi",    425)
    fatigue_subplot(f[:,0], f[:,6],   f_avg[:,6], f_ref[:,6],  f_max[:,6],  sr_mg_log[:,0],  sr_mg_log[:,6],  "tri",    426)
    fatigue_subplot(f[:,0], f[:,7],   f_avg[:,7], f_ref[:,7],  f_max[:,7],  sr_mg_log[:,0],  sr_mg_log[:,7],  "lat",    427)
    fatigue_subplot(f[:,0], f[:,8],   f_avg[:,8], f_ref[:,8],  f_max[:,8],  sr_mg_log[:,0],  sr_mg_log[:,8], "calf",    428)

    # stimulated reps in an exercise
    # plt.figure("exercises" + name)
    # plt.scatter(sr_log[:,0], sr_log[:,1], label="squat" )
    # plt.scatter(sr_log[:,0], sr_log[:,2], label="deadlift" )
    # plt.scatter(sr_log[:,0], sr_log[:,3], label="pullup" )
    # plt.scatter(sr_log[:,0], sr_log[:,4], label="bench" )
    # plt.xlabel("time [day]")
    # plt.title("sr_log_" + name)
    # plt.legend()



def align_mg_pars(mg_pars: dict):
    mat =   np.array([
            mg_pars["quad"],
            mg_pars["ham"],
            mg_pars["abs"],
            mg_pars["pec"],
            mg_pars["bi"],
            mg_pars["tri"],
            mg_pars["lat"],
            mg_pars["calf"]])

    return mat

def compute_sr_d(f0: np.ndarray,
                 pars: dict):

    f_d = align_mg_pars(pars["f_d"])
    f_max = align_mg_pars(pars["f_max"])

    wo_opt = pars["wo_opt"]
    sr_max = 30
    N_ex = 4

    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])

    sr_log = None

    f_cost_values = [np.inf]
    n = 1000
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

    sr_mg_log, f, f_avg = model.compute_model(sr_log=sr_log,
                                        f0=None,
                                        pars=pars)

    sr_d_log = compute_sr_d(f0=f[:, -1],
                             pars=pars)

    sr_d_mg_log, f_d, f_d_avg = model.compute_model(sr_log=sr_d_log,
                                                    f0=None,
                                                    pars=pars)

    # plotting
    #plot_model(sr_log, sr_mg_log, f, f_avg, "data", pars)
    plot_model(sr_d_log, sr_d_mg_log, f_d, f_d_avg, "desired", pars)

    plt.show()
