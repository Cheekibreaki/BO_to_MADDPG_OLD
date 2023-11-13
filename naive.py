# !pip install numpy
# !pip install bayesian-optimization
import json
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from itertools import product
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import matplotlib.pyplot as plt
import random

np.random.seed(2609)
random.seed(2609)
interpreter_path = "E:/Summer Research 2023/DME-DRL Daniel/DME_DRL_CO/venv/Scripts/python.exe "
best_crew_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/BOOF_best_crew.json "
base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map3_1.yaml "
#base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map4_1.yaml "
test_run_config_path = "E:/Summer Research 2023/MADDPG_New/MADDPG/assets/BO_TO_MADDPG/"
worst_performance = float('10000000')


def cost_function(q):
    # Fixed costs for each feature level
    cost_high_high = 80
    cost_high_low = 45
    cost_low_high = 50
    cost_low_low = 10

    if isinstance(q, list):
        total_cost = q[0] * cost_high_high + q[1] * cost_high_low + q[2] * cost_low_high + q[3] * cost_low_low

    else:
        # Calculate the total cost of the crew
        total_cost = q[:, 0] * cost_high_high + q[:, 1] * cost_high_low + q[:, 2] * cost_low_high + q[:,
                                                                                                    3] * cost_low_low

    return total_cost


# define the black box function
def black_box_function(N1, N2, N3, N4):
    # Convert the N values into a list and then into a string for passing to subprocess
    values_as_list = [N1, N2, N3, N4]
    file_path = os.getcwd() + '/priors.json'
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

    return float(result)


# Exploration factor kappa
def dynamic_delta(num_priors, initial_delta, scaling_factor):
    delta = initial_delta / (1 + scaling_factor * num_priors)
    return delta


def sqrt_beta(t=6, delta=0.5, d=4):
    # Confidence Bound for Fixed Budget (CBFB) kauffman et al 2017:
    value = np.sqrt((2 * np.log(t ** (d + 2) * np.pi ** 2 / (3 * delta))) / t)
    return value


# Problem Hyperparameters
budget = 20

if __name__ == "__main__":
    # Generate discrete and linear space
    # Define the search space for the categorical variables N1, N2, N3, and N4
    N_space = [0, 1, 2, 3]

    # Create grid of discrete points in the search space
    grid_points = []
    for N1, N2, N3, N4 in product(N_space, repeat=4):
        grid_points.append([N1, N2, N3, N4])
    grid_points = np.array(grid_points)
    grid_points = grid_points[1:]
    # Initial Priors

    # 1
    # priors = [
    #     {'N1': 2, 'N2': 0, 'N3': 0, 'N4': 3, 'target': black_box_function(2, 0, 0, 3) + cost_function([2, 0, 0, 3])},
    #     # Prior 1
    #     {'N1': 0, 'N2': 3, 'N3': 3, 'N4': 0, 'target': black_box_function(0, 3, 3, 0) + cost_function([0, 3, 3, 0])},
    #     # Prior 2
    #     {'N1': 1, 'N2': 1, 'N3': 1, 'N4': 2, 'target': black_box_function(1, 1, 1, 2) + cost_function([1, 1, 1, 2])},
    #     # Prior 3
    #     {'N1': 3, 'N2': 2, 'N3': 2, 'N4': 1, 'target': black_box_function(3, 2, 2, 1) + cost_function([3, 2, 2, 1])},
    #     # prior 4
    #     {'N1': 3, 'N2': 1, 'N3': 3, 'N4': 1, 'target': black_box_function(3, 1, 3, 1) + cost_function([3, 1, 3, 1])},
    #     # prior 5
    # ]

    # #2
    priors = [
            {'N1': 0, 'N2': 1, 'N3':1, 'N4':3, 'target': black_box_function(0, 1, 1, 3)+cost_function([0, 1, 1, 3])},   # Prior 1
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': black_box_function(2, 2, 2, 1)+cost_function([2, 2, 2, 1])},   # Prior 2
            {'N1': 3, 'N2': 0, 'N3':0, 'N4':2, 'target': black_box_function(3, 0, 0, 2)+cost_function([3, 0, 0, 2])},   # Prior 3
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': black_box_function(1, 3, 3, 0)+cost_function([1, 3, 3, 0])},   #prior 4
            {'N1': 1, 'N2': 0, 'N3':2, 'N4':0, 'target': black_box_function(1, 0, 2, 0)+cost_function([1, 0, 2, 0])},   #prior 5
        ]

    #3 
    # priors = [
    #         {'N1': 3, 'N2': 3, 'N3':2, 'N4':1, 'target': black_box_function(3, 3, 2, 1)+cost_function([3, 3, 2, 1])},   # Prior 1
    #         {'N1': 1, 'N2': 0, 'N3':0, 'N4':3, 'target': black_box_function(1, 0, 0, 3)+cost_function([1, 0, 0, 3])},   # Prior 2
    #         {'N1': 0, 'N2': 2, 'N3':3, 'N4':0, 'target': black_box_function(0, 2, 3, 0)+cost_function([0, 2, 3, 0])},   # Prior 3
    #         {'N1': 2, 'N2': 1, 'N3':1, 'N4':2, 'target': black_box_function(2, 1, 1, 2)+cost_function([2, 1, 1, 2])},   #prior 4
    #         {'N1': 2, 'N2': 2, 'N3':0, 'N4':2, 'target': black_box_function(2, 2, 0, 2)+cost_function([2, 2, 0, 2])},   #prior 5
    #     ]

    # #4
    # priors = [
    #         {'N1': 1, 'N2': 3, 'N3':1, 'N4':2, 'target': black_box_function(1, 3, 1, 2)+cost_function([1, 3, 1, 2])},   # Prior 1
    #         {'N1': 2, 'N2': 0, 'N3':2, 'N4':1, 'target': black_box_function(2, 0, 2, 1)+cost_function([2, 0, 2, 1])},   # Prior 2
    #         {'N1': 3, 'N2': 2, 'N3':0, 'N4':3, 'target': black_box_function(3, 2, 0, 3)+cost_function([3, 2, 0, 3])},   # Prior 3
    #         {'N1': 0, 'N2': 1, 'N3':3, 'N4':0, 'target': black_box_function(0, 1, 3, 0)+cost_function([0, 1, 3, 0])},   #prior 4
    #         {'N1': 0, 'N2': 2, 'N3':2, 'N4':0, 'target': black_box_function(0, 2, 2, 0)+cost_function([0, 2, 2, 0])},   #prior 5
    #     ]

    #5
    # priors = [
    #     {'N1': 1, 'N2': 3, 'N3': 3, 'N4': 0, 'target': black_box_function(1, 3, 3, 0) + cost_function([1, 3, 3, 0])},
    #     # Prior 1
    #     {'N1': 3, 'N2': 1, 'N3': 0, 'N4': 3, 'target': black_box_function(3, 1, 0, 3) + cost_function([3, 1, 0, 3])},
    #     # Prior 2
    #     {'N1': 2, 'N2': 2, 'N3': 2, 'N4': 1, 'target': black_box_function(2, 2, 2, 1) + cost_function([2, 2, 2, 1])},
    #     # Prior 3
    #     {'N1': 0, 'N2': 0, 'N3': 1, 'N4': 2, 'target': black_box_function(0, 0, 1, 2) + cost_function([0, 0, 1, 2])},
    #     # prior 4
    #     {'N1': 0, 'N2': 2, 'N3': 0, 'N4': 2, 'target': black_box_function(0, 2, 0, 2) + cost_function([0, 2, 0, 2])},
    #     # prior 5
    # ]

    count = 1
    while count <= budget:
        print('Iteration: ', count)

        # Dinamic Exploration parameter
        ddelta = dynamic_delta(len(priors) + 1, 0.6, 1)
        kappa = sqrt_beta(t=len(priors) + 1, delta=ddelta)  # UCB kappa parameter/ t should be number of priors + 1

        # Initialize the Gaussian process regressor
        kernel = RBF(length_scale=1.0)

        regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                             normalize_y=True,
                                             n_restarts_optimizer=5,
                                             random_state=13)

        # Prepare the data for Gaussian process regression
        P = np.array([[p['N1'], p['N2'], p['N3'], p['N4']] for p in priors])
        Z = np.array([p['target'] for p in priors])

        # Fit the Gaussian process regressor
        regressor.fit(P, Z)

        mu, sigma = regressor.predict(grid_points, return_std=True)
        LCB = mu - kappa * sigma
        best_index = np.argmin(LCB)
        # Retrieve the best solution and its corresponding objective values
        best_solution = grid_points[best_index]
        best_objectives = LCB[best_index]

        # Evaluate the black-box function for the best_solution
        # if np.array_equal(best_solution, [0, 0, 0, 0]):
        #    best_prior = {'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'target': worst_performance}
        #    best_performance = worst_performance
        # else:
        best_performance = black_box_function(best_solution[0], best_solution[1], best_solution[2], best_solution[3])
        best_cost = cost_function(list(best_solution))

        # Append the best_solution and its performance to the list of priors
        best_prior = {
            'N1': int(best_solution[0]),
            'N2': int(best_solution[1]),
            'N3': int(best_solution[2]),
            'N4': int(best_solution[3]),
            'target': best_performance + best_cost
        }

        priors.append(best_prior)
        print("Point suggestion : {}, value: {}".format(best_solution, best_performance + best_cost))
        count += 1

    visited_utility = np.array([p['target'] for p in priors])
    visited_crews = np.array([[p['N1'], p['N2'], p['N3'], p['N4']] for p in priors])
    visited_cost = np.array([cost_function(list(visited_crew)) for visited_crew in visited_crews])
    best_visited_utility = np.argmin(visited_utility)
    best_crew = visited_crews[best_visited_utility]

    print("visited_crews", visited_crews)
    print('visited_performance', visited_utility - visited_cost)
    print("visited_cost", visited_cost)

    print("Best point suggestion : {}, value: {}".format(best_crew, np.min(visited_utility)))

    # Convert visited_crews array to a list of strings to use as x-axis ticks
    x_data = [' '.join(map(str, crew)) for crew in visited_crews]

    # Create an array of indices for x-axis positioning
    x_indices = np.arange(len(x_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x_indices, visited_utility, visited_cost, c='r', marker='o')

    # Setting x-ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_data, rotation=45, fontsize=8)  # Rotate for readability

    ax.set_xlabel('Crew Combinations')
    ax.set_ylabel('Total Time')
    ax.set_zlabel('Cost')
    plt.ylim(0, 2000);
    plt.tight_layout()
    plt.title("Naive Result")
    visited_utility = visited_utility
    print('visited_utility', visited_utility)
    best_visited_utility = np.argmin(visited_utility)
    best_crew = visited_crews[best_visited_utility]
    print("Best point suggestion : {}, iteration {}, value: {}".format(best_crew, best_visited_utility,
                                                                       np.min(visited_utility)))
    plt.show()

    # Convert visited_crews array to a list of strings to use as x-axis ticks
    x_data = [' '.join(map(str, crew)) for crew in visited_crews]
    x_data = x_data[:20]
    # Create an array of indices for x-axis positioning
    x_indices = np.arange(len(x_data))

    # First 2D graph: Crew Combinations vs Performance + Cost
    plt.figure(figsize=(10, 5))
    plt.bar(x_indices, visited_utility - visited_cost, color='blue')
    plt.xticks(x_indices, x_data, rotation=45, fontsize=8)
    plt.ylabel('Total Time')
    plt.xlabel('Crew Combinations')
    plt.tight_layout()
    plt.title("Naive Total Time")
    plt.ylim(0, 2000)
    plt.show()

    # Second 2D graph: Crew Combinations vs Cost
    plt.figure(figsize=(10, 5))
    plt.bar(x_indices, visited_cost, color='green')
    plt.xticks(x_indices, x_data, rotation=45, fontsize=8)
    plt.ylabel('Cost')
    plt.xlabel('Crew Combinations')
    plt.title("Naive Cost")
    plt.tight_layout()
    plt.show()