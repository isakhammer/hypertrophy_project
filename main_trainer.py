import numpy as np
from matplotlib import pyplot as plt
import random
import json
import configparser
import os


# class MuscleGroup:

#     def __init__(self, recovery_rate, stimulated_reps):
#         self.current_day = 0
#         self.prev_workout_day = 0
#         self.prev_fatigue = 0
#         self.recovery_rate = recovery_rate
#         self.sr_log = None

#     def update(self, stimulated_reps):
#         self.prev_workout_day = self.current_day
#         self.prev_fatigue += self.compute_fatigue() + stimulated_reps

#     def compute_fatigue(self):
#         return fatigue

def fatigue( pars, sr_log):

    """
    input:
    pars:     dict of all parameters
    sr_log:   imported stimulated reps [day, squat, deadlift, pullup, bench]

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


    sr_muscle_group_log = compute_sr_mg_log(sr_log, pars)


    # time interval
    N = 100
    t_start = 0
    t_end = np.amax(sr_muscle_group_log[:,0])
    t_interval = np.linspace(t_start, t_end, N)

    # fatigue
    n_groups = 8
    f = np.zeros(time_iterations, n_groups + 1)
    f[:,0] = t_interval

    # Workout number j
    j_sr = 0
    f_j_sr = np.zeros(n_groups)
    t_j_sr = 0


    for i in range(time_iterations):
        t_i = t_interval[i]

        # saving fatigue at previous workout
        if j_sr < sr_muscle_group_log.shape[0] and sr_muscle_group_log[j_sr, 0] < t_i:
            f[i, 1:] += sr_muscle_group_log[j_sr, 1:]
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
    sr_log:   imported stimulated reps [day, squat, deadlift, pullup, bench]

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


def load_params( file_paths ):

    # load vehicle parameter file into a "pars" dict
    parser = configparser.ConfigParser()

    if not parser.read(file_paths["params"]):
        raise ValueError('Specified config file does not exist or is empty!')

    pars = {}
    rec_rates = json.loads(parser.get('RECOVERY_OPTIONS', 'recovery_rates'))
    pars["rec_rates"] = rec_rates

    ex = {}
    ex['squat'] = json.loads(parser.get('EXERCISE_OPTIONS', 'squat'))
    ex['deadlift'] = json.loads(parser.get('EXERCISE_OPTIONS', 'deadlift'))
    ex['bench'] = json.loads(parser.get('EXERCISE_OPTIONS', 'bench'))
    ex['pullup'] = json.loads(parser.get('EXERCISE_OPTIONS', 'pullup'))

    pars["ex"] = ex
    return pars

def import_log( file_paths: dict) -> np.ndarray:
    """
    Imports file path of raw data with csv-format containing:
    #day;squat;deadlift;pullup;bench

    Outputs:
    sr_log:   imported stimulated reps [day, squat, deadlift, pullup, bench]
    """

    def skipper(fname):
        """
        Function for skipping header in csv

        """
        with open(fname) as fin:
            no_comments = (line for line in fin if not line.lstrip().startswith('#'))
            next(no_comments, None) # skip header
            for row in no_comments:
                yield row


    # load data from csv file
    csv_data_tmp = np.loadtxt(skipper(file_paths["sr_log"]), delimiter=';')

    # get coords and track widths out of array
    if np.shape(csv_data_tmp)[1] == 5:
        day         = csv_data_tmp[:,0]
        squat       = csv_data_tmp[:,1]
        deadlift    = csv_data_tmp[:,2]
        pullup      = csv_data_tmp[:,3]
        bench       = csv_data_tmp[:,4]

    else:
        raise IOError(file_paths["sr_log"] + " cannot be read!")

    # assemble to a single array
    sr_log = np.column_stack((day,
                                squat,
                                deadlift,
                                pullup,
                                bench
                                ))

    return sr_log


def initalize_muscle_groups(pars):
    recovery_rates = pars["recovery_rates"]
    muscle_groups = {}
    muscle_groups["quad"]   = MuscleGroup(     recovery_rates["quad"])
    muscle_groups["ham"]    = MuscleGroup(     recovery_rates["ham"])
    muscle_groups["pec"]    = MuscleGroup(     recovery_rates["pec"])
    muscle_groups["abs"]    = MuscleGroup(     recovery_rates["abs"])
    muscle_groups["bi"]     = MuscleGroup(     recovery_rates["bi"])
    muscle_groups["tri"]    = MuscleGroup(     recovery_rates["tri"])
    muscle_groups["lat"]    = MuscleGroup(     recovery_rates["lat"])
    muscle_groups["calf"]   = MuscleGroup(     recovery_rates["calf"])

    return muscle_groups







if __name__=="__main__":
    # load parameters
    file_paths = {}
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
    file_paths["params"] = os.path.join(file_paths["module"], "params.ini")
    file_paths["sr_log"] = os.path.join(file_paths["module"],"sr_log.csv")

    pars = load_params(file_paths)
    sr_log = import_log(file_paths)
    fatigue(pars, sr_log)















