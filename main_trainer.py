import numpy as np
from matplotlib import pyplot as plt
import random



class MuscleGroup:

    def __init__(self, recovery_rate, start_fatigue):
        self.day = 0
        self.prev_fatigue = 0
        self.recovery_rate = recovery_rate

    def update(self, stimulated_reps):
        self.day +=1
        self.prev_fatigue += self.compute_fatigue() + stimulated_reps

    def compute_fatigue(self):
        fatigue = self.prev_fatigue*np.exp(-self.days*self.recovery_rate)
        return fatigue


def main():

    # Initialize muscle groups
    muscle_groups = {

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








