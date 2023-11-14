# this is the implementation of Discrete BO and its interface

# !pip install Gpy
# install Gpy of 1.10.0

import time
import numpy as np
import math
import random
import json
import os
import subprocess

from Space import Space
from BayesianOptimizer import BayesianOptimizer
from BayesianOptimizer import Xobs, Yobs

np.random.seed(2609)
random.seed(2609)
worst_performance = float('10000000')
# interpreter_path = "C:/Users/Daniel Yin/AppData/Local/Programs/Python/Python39/python.exe"
interpreter_path = "E:/Summer Research 2023/DME-DRL Daniel/DME_DRL_CO/venv/Scripts/python.exe "
best_crew_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/D_BO/D_BO_best_crew.json "
base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map4_1.yaml "
test_run_config_path = "E:/Summer Research 2023/MADDPG_New/MADDPG/assets/BO_TO_MADDPG/"


calculated_dict = {}
# Dictionary to store satisfaction values for each unique crew
satisfaction_dict = {}

def calculate_cost(X):
    # Fixed costs for each feature level
    q_low_low = X[3]
    q_high_low = X[1]
    q_low_high = X[2]
    q_high_high = X[0]
    cost_low_low = 10
    cost_high_low = 45
    cost_low_high = 50
    cost_high_high = 80

    # Calculate the total cost of the crew
    total_cost = q_low_low * cost_low_low + q_high_low * cost_high_low + q_low_high * cost_low_high + q_high_high * cost_high_high
    return float(total_cost)


class discreteBranin:
    # def __init__(self):
    #     self.numCalls = 0
    #     self.input_dim = 2
    #     self.bounds = [(-5, 10),(0, 15)]
    #     self.fmin = 0.497910
    #     self.min = [-3,12]
    #     self.ismax = -1
    #     self.name = 'DiscreteBranin'
    #     self.discreteIdx = [0,1]
    #     self.categoricalIdx = [0, 1]
    #     self.spaces = [Space(-5, 10, True), Space(0, 15, True)]

    def __init__(self):
        self.numCalls = 0
        self.input_dim = 4
        self.bounds = [(0, 3), (0, 3), (0, 3), (0, 3)]
        # self.fmin = 0.497910
        # self.min = [-3,12]
        self.ismax = 1
        self.name = 'DiscreteBranin'
        self.discreteIdx = [0, 1, 2, 3]
        self.categoricalIdx = [0]
        self.spaces = [Space(0, 3, True), Space(0, 3, True), Space(0, 3, True), Space(0, 3, True)]

    def black_box_function(self, X):
        self.numCalls += 1
        values_as_list = []
        for xi in X:
            if xi != int(xi):
                print(xi)
                raise Exception("Invalid input")
            # Convert the N values into a list and then into a string for passing to subprocess
            values_as_list.append(xi)

        file_path = os.getcwd() + '/D_BO_best_crew.json'
        # print("values_as_list", values_as_list)

        crew = np.array(values_as_list)
        # Write the data to the JSON file
        with open(file_path, 'w') as file:
            json.dump(crew.tolist(), file)
        try:
            result = subprocess.run([interpreter_path, 'exploration_initializer.py',
                                     file_path, base_config_path, test_run_config_path], check=True, cwd=os.getcwd(),
                                    stdout=subprocess.PIPE, text=True, encoding='utf-8')
            result = result.stdout.splitlines()[-1]  # The standard output of the subprocess
            # Now 'result' is properly defined within the try block
        except subprocess.CalledProcessError as e:
            print(f"Error running exploration script_path: {e}")
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        return float(result) + calculate_cost(crew)

    def call_initializer(self, solution):
        # Specify the file path
        file_path = os.getcwd() + '/D_BO_best_crew.json'

        crew = np.array(solution)
        # Write the data to the JSON file
        with open(file_path, 'w') as file:
            json.dump(crew.tolist(), file)
        try:
            result = subprocess.run([interpreter_path, 'exploration_initializer.py',
                                     file_path, base_config_path, test_run_config_path], check=True, cwd=os.getcwd(),
                                    stdout=subprocess.PIPE, text=True, encoding='utf-8')
            result = result.stdout.splitlines()[-1]  # The standard output of the subprocess
            # Now 'result' is properly defined within the try block
        except subprocess.CalledProcessError as e:
            print(f"Error running exploration script_path: {e}")
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        return float(result) + float(calculate_cost(crew))

    # define the black box function
    def _interfunc(self, X):
        # TODO: change func to call subprocess; this include init priors and upcoming best crews
        # Fixed costs for each feature level
        return self.call_initializer(X)

    def func(self, X):
        self.numCalls += 1
        for xi in X:
            if xi != int(xi):
                print(xi)
                raise Exception("Invalid input")
        return self._interfunc(X)

    def randInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = int(np.random.uniform(i[0], i[1]+1))

            # print("randInBounds",tmpRand)
            rand.append(tmpRand)
        return rand

    def randUniformInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = np.random.uniform(i[0], i[1])

            # print("randUniformInBounds",tmpRand)
            rand.append(tmpRand)
        return rand

    def normalize(self, x):
        val = []
        for i in range(0, self.input_dim):
            val.append((x[i] - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0]))
        return val

    def denormalize(self, x):
        val = []
        for i in range(0, self.input_dim):
            val.append(1.0 * (x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]))
        return val


# DISCRETE BO
myfunction = discreteBranin()

num_test = 1
for i in range(num_test):
    print(type(Xobs))
    BO = BayesianOptimizer(myfunction)
    BOstart_time = time.time()
    bestBO, box, boy, ite = BO.run(method="DiscreteBO")
    BOstop_time = time.time()
    ite = ite-1
    print("Iter 0:", " Discrete BO x: ", bestBO, " y:", -1.0*myfunction.func(bestBO), " ite:", ite, " time: --- %s seconds ---" % (BOstop_time - BOstart_time))
    # print(type(bestBO))
    BO.resetBO()

    print(type(Xobs))
