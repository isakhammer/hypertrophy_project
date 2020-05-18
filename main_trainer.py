import numpy as np
from matplotlib import pyplot as plt
import random
import json
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


def main():

    # load vehicle parameter file into a "pars" dict
    parser = configparser.ConfigParser()
    pars = {}

    # load parameters
    file_paths = {}
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
    file_paths["params"] = os.path.join(file_paths["module"], "params.ini")

    if not parser.read(file_paths["params"]):
        raise ValueError('Specified config file does not exist or is empty!')

    recovery_rate = json.loads(parser.get('RECOVERY_OPTIONS', 'recovery_rate'))

    muscle_info = {
        "quadriceps":
    }
    # Initialize muscle groups
    muscle_groups = {
        "legs"
    }

    # Initialize exercises and how much it affect muscle groups
    # Should be loaded from json file
    exercises = {

    }

    # Initialize muscle groups from csv
    muscle_groups = {

    }


if __name__=="__main__":
    main()

    print("Hello")








