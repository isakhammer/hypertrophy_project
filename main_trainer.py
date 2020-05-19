import numpy as np
from matplotlib import pyplot as plt
import random
import json
import configparser
import os


class MuscleGroup:

    def __init__(self, recovery_rate, start_fatigue):
        self.current_day = 0
        self.prev_workout_day = 0
        self.prev_fatigue = 0
        self.recovery_rate = recovery_rate

    def update(self, stimulated_reps):
        self.prev_workout_day = self.current_day
        self.prev_fatigue += self.compute_fatigue() + stimulated_reps

    def compute_fatigue(self):
        fatigue = self.prev_fatigue*np.exp(-(self.current_day - self.prev_workout_day)*self.recovery_rate)
        return fatigue


def load_params( file_paths ):

    # load vehicle parameter file into a "pars" dict
    parser = configparser.ConfigParser()

    if not parser.read(file_paths["params"]):
        raise ValueError('Specified config file does not exist or is empty!')

    pars = {}
    recovery_rate = json.loads(parser.get('RECOVERY_OPTIONS', 'recovery_rate'))

    pars["recovery_rate"] = recovery_rate

    exercises = {}
    exercises['squat'] = json.loads(parser.get('EXERCISE_OPTIONS', 'squat'))
    exercises['deadlift'] = json.loads(parser.get('EXERCISE_OPTIONS', 'deadlift'))
    exercises['bench'] = json.loads(parser.get('EXERCISE_OPTIONS', 'bench'))
    exercises['pullup'] = json.loads(parser.get('EXERCISE_OPTIONS', 'pullup'))

    pars["exercises"] = exercises
    return pars

def import_log( file_paths: dict) -> np.ndarray:
    """
    Imports file path of raw data with csv-format containing:
    #day;squat;deadlift;pullup;bench

    Outputs:
    data:   imported rir data [day, squat, deadlift, pullup, bench]
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
    csv_data_tmp = np.loadtxt(skipper(file_paths["rir_log"]), delimiter=';')

    # get coords and track widths out of array
    if np.shape(csv_data_tmp)[1] == 5:
        day         = csv_data_tmp[:,0]
        squat       = csv_data_tmp[:,1]
        deadlift    = csv_data_tmp[:,2]
        pullup      = csv_data_tmp[:,3]
        bench       = csv_data_tmp[:,4]

    else:
        raise IOError(file_paths["rir_log"] + " cannot be read!")

    # assemble to a single array
    rir_data = np.column_stack((day,
                                squat,
                                deadlift,
                                pullup,
                                bench
                                ))

    return rir_data




if __name__=="__main__":
    def load_file_paths():
        # load parameters
        file_paths = {}
        file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
        file_paths["params"] = os.path.join(file_paths["module"], "params.ini")
        file_paths["rir_log"] = os.path.join(file_paths["module"],"rir_log.csv")
        return file_paths

    file_paths = load_file_paths()
    pars = load_params(file_paths)
    rir_data = import_log(file_paths)
    print(rir_data)













