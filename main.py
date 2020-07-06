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
    N_ex = len(pars["ex_names"])
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
    pars["f_star"]         = json.loads(parser.get('FATIGUE_OPTIONS', 'f_star'))
    pars["rec_rates"]   = json.loads(parser.get('FATIGUE_OPTIONS', 'rec_rates'))

    pars["mg_names"] = json.loads(parser.get('GENERAL', 'mg_names'))
    pars["ex_names"] = json.loads(parser.get('GENERAL', 'ex_names'))

    ex = {}
    for ex_name in pars["ex_names"]:
        ex[ex_name] = json.loads(parser.get('EXERCISE_OPTIONS', ex_name))

    pars["ex"] = ex

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



def plot_model(     sr_ex_log:    np.ndarray,
                    sr_mg_log: np.ndarray,
                    f_log:         np.ndarray,
                    f_avg_log:     np.ndarray,
                    f_star:    np.ndarray,
                    f_max:     np.ndarray,
                    name:      str,
                    pars:      dict):
    """
    Input:
        sr_ex_log:  [t, sr_ex0, sr_ex1, ... ]       (N_t, N_mg + 1)
        sr_mg_log:  [t, sr_mg0, sr_mg1, ... ]       (N_t, N_mg + 1)
        f_log:      [t, f_mg0, f_mg1, ... ]         (N_t, N_mg + 1)
        f_avg_log:  [t, f_mg0, f_mg1, ... ]         (N_t, N_mg + 1)
        f_star:     [f_mg, f_mg1, ... ]             (N_mg, )
        f_max:      [f_mg, f_mg1, ... ]             (N_mg, )


    name:       name of plot
    pars:       all imported parameters

    """



    plt.rcParams['axes.labelsize'] = 10.0
    plt.rcParams['axes.titlesize'] = 11.0
    plt.rcParams['legend.fontsize'] = 10.0
    #plt.rcParams['figure.figsize'] = 25 / 2.54, 20 / 2.54
    plt.rcParams["figure.figsize"] = [16,9]


    # plot muscle fatigue
    def fatigue_subplot(f_t,
                        f_mg,
                        f_avg_mg,
                        f_ref_mg,
                        f_max_mg,
                        sr_mg_t,
                        sr_mg,
                        title: str,
                        nrows: int,
                        ncols: int,
                        index: int):


        plt.subplot(nrows, ncols, index)
        plt.plot(f_t,       f_mg,       label="f" )
        plt.plot(f_t,       f_ref_mg,   label="f_ref" )
        plt.plot(f_t,       f_max_mg,   label="f_max" )
        plt.plot(f_t,       f_avg_mg,   label="f_avg" )
        plt.scatter(sr_mg_t,   sr_mg,   label="sr_mg", s=7, color="black")
        plt.ylim((0,25))
        plt.title(title)
        plt.grid()
        plt.legend()

    plt.figure(name)
    plt.clf()

    # Computing nrows and ncols in pyplot subplots
    mg_names = pars["mg_names"]

    N_mg = len(mg_names)
    ncols = 2
    nrows = None
    if N_mg%ncols == 0:
        nrows = N_mg//ncols
    else:
        nrows = N_mg//ncols + 1

    # Discretize f_star and f_max
    def init_log(t, f):
        """
        Function to initalize mg parameters in correct format
        Input:
            t:      [t0, t1, ... ] (N_t, )  array for time
            f: [f_mg, f_mg1, ... ] (N_mg, )

        """
        ones = np.ones(t.shape)
        mat = []
        mat.append(t)
        for i in range(f.shape[0]):
            mat.append(ones*f[i])

        mat = np.array(mat).T

        return mat

    f_t = f_log[:,0]
    f_star_log      = init_log(f_t, f_star)
    f_max_log       = init_log(f_t, f_max)

    # Plot fatigue for each muscle group
    for i in range(1, N_mg + 1):
        fatigue_subplot(f_t, f_log[:,i],   f_avg_log[:,i], f_star_log[:,i],  f_max_log[:,i],  sr_mg_log[:,0],  sr_mg_log[:,i], mg_names[i-1],    nrows, ncols, i)

    # plot stimulated reps for exercises
    def ex_subplot(   sr_t: np.ndarray,
                      sr_ex: np.ndarray,
                      title:  str,
                      nrows:  int,
                      ncols:  int,
                      index:  int):
        plt.subplot(nrows, ncols, index)
        plt.plot(sr_t, sr_ex, label="stimulated reps")
        plt.grid()
        plt.title(title)
        plt.legend()

    plt.figure("exercises_" + name)
    ex_names = pars["ex_names"]
    N_ex = len(ex_names)
    ncols = 2
    nrows = None
    if N_ex%ncols == 0:
        nrows = N_ex//ncols
    else:
        nrows = N_ex//ncols + 1

    for i in range(1, N_ex + 1):
        ex_subplot(sr_ex_log[:,0], sr_ex_log[:,i], ex_names[i-1], nrows, ncols, i)



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
    sr_ex_log = import_log(file_paths)

    sr_mg_log, f_log, f_avg_log = model.compute_model(sr_ex_log=sr_ex_log,
                                        f0=None,
                                        pars=pars)

    print(sr_ex_log)
    f_star = align_mg_pars(pars["f_star"])
    f_max = align_mg_pars(pars["f_max"])
    f0 = f_log[-1, 1:]

    method = "pso"

    if method =="bfo":
        sr_d_ex_log = bfo.brute_force_optimization(f0=f0,
                                                f_d=f_star,
                                                f_max=f_max,
                                                pars=pars)
    elif method =="pso":
        sr_d_ex_log = pso.particle_swarm_optimization( f0=f0,
                                                    f_d=f_star,
                                                    f_max=f_max,
                                                    pars=pars)

    sr_d_mg_log, f_d_log, f_d_avg_log = model.compute_model(sr_ex_log=sr_d_ex_log,
                                                    f0=f0,
                                                    pars=pars)

    # Plot
    plot_model(sr_ex_log=sr_d_ex_log,
               sr_mg_log=sr_d_mg_log,
               f_log=f_d_log,
               f_avg_log=f_d_avg_log,
               f_star=f_star,
               f_max=f_max,
               name="desired",
               pars=pars)

    plot_model(sr_ex_log=sr_d_ex_log,
               sr_mg_log=sr_d_mg_log,
               f_log=f_d_log,
               f_avg_log=f_d_avg_log,
               f_star=f_star,
               f_max=f_max,
               name="log",
               pars=pars)


    plt.show()
