import numpy as np
from matplotlib import pyplot as plt
from src import model
from src import brute_force_optimization as bfo
from src import particle_swarm_optimization as pso
import random
import json
import configparser
import os

def feasability_check(pars: dict):

    N_mg = len(pars["mg_names"])
    N_ex = pars["N_ex"]
    ex_pars = pars["ex"]

    for ex_name in ex_pars:
        ex_par = ex_pars[ex_name]
        if N_mg != len(ex_par):
            print("ERROR", ex_name, " is not feasible. mg names does not match")
            exit()




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

    pars["mg_names"] = json.loads(parser.get('GENERAL', 'mg_names'))
    pars["ex_names"] = json.loads(parser.get('GENERAL', 'ex_names'))

    ex = {}
    for ex_name in pars["ex_names"]:
        ex[ex_name] = json.loads(parser.get('EXERCISE_OPTIONS', ex_name))

    pars["ex"] = ex
    pars["N_mg"] = json.loads(parser.get('GENERAL', 'N_mg'))
    pars["N_ex"] = json.loads(parser.get('GENERAL', 'N_ex'))

    feasability_check(pars)
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
    csv_data = np.loadtxt(skipper(file_paths["sr_log"], header=True), delimiter=';')

    return csv_data



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
        f_mg_pars = align_mg_pars(f_pars)

        ones = np.ones(t.shape)
        mat = []
        mat.append(t)
        for i in range(f_mg_pars.shape[0]):
            mat.append(ones*f_mg_pars[i])

        mat = np.array(mat).T

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

    plt.figure("exercises_" + name)
    plt.subplot(411)
    plt.plot(sr_log[:,0], sr_log[:,1], label="squat" )
    plt.legend()
    plt.subplot(412)
    plt.plot(sr_log[:,0], sr_log[:,2], label="deadlift" )
    plt.legend()
    plt.subplot(413)
    plt.plot(sr_log[:,0], sr_log[:,3], label="pullup" )
    plt.legend()
    plt.subplot(414)
    plt.plot(sr_log[:,0], sr_log[:,4], label="bench" )
    plt.legend()



def align_mg_pars(mg_pars: dict):
    mg = []
    for mg_name in mg_pars:
        mg.append(mg_pars[mg_name])
    mg = np.array(mg)
    return mg


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

    f_d = align_mg_pars(pars["f_d"])
    f_max = align_mg_pars(pars["f_max"])

    method = "pso"

    if method =="bfo":
        sr_d_log = bfo.brute_force_optimization(f0=f[:, -1],
                                                f_d=f_d,
                                                f_max=f_max,
                                                pars=pars)
    elif method =="pso":
        sr_d_log = pso.particle_swarm_optimization( f0=f[:, -1],
                                                    f_d=f_d,
                                                    f_max=f_max,
                                                    pars=pars)

    sr_d_mg_log, f_d, f_d_avg = model.compute_model(sr_log=sr_d_log,
                                                    f0=None,
                                                    pars=pars)

    # plotting
    #plot_model(sr_log, sr_mg_log, f, f_avg, "data", pars)
    plot_model(sr_d_log, sr_d_mg_log, f_d, f_d_avg, "desired", pars)

    plt.show()
