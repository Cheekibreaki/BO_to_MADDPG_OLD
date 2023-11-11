# BENCHMARK 2
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
# base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map4_1.yaml "
base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map3_1.yaml "
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

# subprocess call
def call_initializer(solution):
    # Specify the file path
    file_path = os.getcwd() + '/BOFD_best_crew.json'

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

    new_data = {
        'N1': solution[0],
        'N2': solution[1],
        'N3': solution[2],
        'N4': solution[3],
        'target': result  # Adjust the arguments as necessary
    }

    return new_data, result




def get_random_satisfaction(crew):
    # Check if the satisfaction for the crew is already in the dictionary
    if tuple(crew) in satisfaction_dict:
        return satisfaction_dict[tuple(crew)]
    else:
        # Generate a random satisfaction value between 0 and 1000
        satisfaction = random.uniform(satisfaction_lower_bound, satisfaction_upper_bound)
        # Store the satisfaction value in the dictionary for future use
        satisfaction_dict[tuple(crew)] = satisfaction
        return satisfaction


def generate_all_crews():
    crews = []
    # Loop through all possible combinations of quantities for each feature level
    for q_low_low in range(4):
        for q_high_low in range(4):
            for q_low_high in range(4):
                for q_high_high in range(4):
                    crew = [q_low_low, q_high_low, q_low_high, q_high_high]
                    crews.append(crew)
    return crews


def generate_satisfaction_for_all_crews():
    crews = generate_all_crews()
    for crew in crews:
        get_random_satisfaction(crew)


def max_satisfaction():
    crews = generate_all_crews()
    max_value = float('inf')  # Initialize to negative infinity
    best_crew = None
    for crew in crews:
        satisfaction_utility = black_box_function(crew[0], crew[1], crew[2], crew[3]) + cost_function(crew)
        if satisfaction_utility < max_value:
            max_value = satisfaction_utility
            best_crew = crew

    return max_value, best_crew


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

    return result

# Exploration factor kappa
def dynamic_delta(num_priors, initial_delta, scaling_factor):
    delta = initial_delta / (1 + scaling_factor * num_priors)
    return delta


def sqrt_beta(t=6, delta=0.5, d=4):
    # Confidence Bound for Fixed Budget (CBFB) kauffman et al 2017:
    value = np.sqrt((2 * np.log(t ** (d + 2) * np.pi ** 2 / (3 * delta))) / t)
    return value


# Define a custom problem for NSGA-II
class CostProblem(Problem):
    # to do: change input to receive distribution
    def __init__(self, regressor, regressor2, kappa):
        super().__init__(n_var=4, n_obj=2, n_constr=1, xl=np.array([0, 0, 0, 0]), xu=np.array([3, 3, 3, 3]))
        self.regressor = regressor
        self.regressor2 = regressor2
        self.kappa = kappa

    def _evaluate(self, X, out, *args, **kwargs):
        # Add a constraint to ensure that the sum is greater than 0
        constraint = 1 - np.sum(X, axis=1)

        # to do: call performance from f1
        mu, sigma = self.regressor.predict(X, return_std=True)
        # LCB score as performance
        f1 = mu - self.kappa * sigma
        # Cost function:
        mu2, sigma2 = self.regressor2.predict(X, return_std=True)
        f2 = mu2 - self.kappa * sigma2

        out["F"] = np.column_stack([f1, f2])
        out["G"] = constraint

# Problem Hiperparameters
budget = 20
satisfaction_dict = {}  # Dictionary to store satisfaction values for each unique crew
satisfaction_lower_bound = 0
satisfaction_upper_bound = 750

if __name__ == "__main__":

    # Generate satisfaction values for all 1364 different crews
    generate_satisfaction_for_all_crews()

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
    '''
    
    # 1
    priors = [
         {'N1': 2, 'N2': 0, 'N3': 0, 'N4': 3, 'target': black_box_function(2, 0, 0, 3)},  # Prior 1
         {'N1': 0, 'N2': 3, 'N3': 3, 'N4': 0, 'target': black_box_function(0, 3, 3, 0)},  # Prior 2
         {'N1': 1, 'N2': 1, 'N3': 1, 'N4': 2, 'target': black_box_function(1, 1, 1, 2)},  # Prior 3
         {'N1': 3, 'N2': 2, 'N3': 2, 'N4': 1, 'target': black_box_function(3, 2, 2, 1)},  # prior 4
         {'N1': 3, 'N2': 1, 'N3': 3, 'N4': 1, 'target': black_box_function(3, 1, 3, 1)},  # prior 5
     ]

    priors2 = [
         {'N1': 2, 'N2': 0, 'N3': 0, 'N4': 3, 'target': cost_function([2, 0, 0, 3])},  # Prior 1
         {'N1': 0, 'N2': 3, 'N3': 3, 'N4': 0, 'target': cost_function([0, 3, 3, 0])},  # Prior 2
         {'N1': 1, 'N2': 1, 'N3': 1, 'N4': 2, 'target': cost_function([1, 1, 1, 2])},  # Prior 3
         {'N1': 3, 'N2': 2, 'N3': 2, 'N4': 1, 'target': cost_function([3, 2, 2, 1])},  # prior 4
         {'N1': 3, 'N2': 1, 'N3': 3, 'N4': 1, 'target': cost_function([3, 1, 3, 1])},  # prior 5
     ]
    
    
    #2

    priors = [
            {'N1': 0, 'N2': 1, 'N3':1, 'N4':3, 'target': black_box_function(0, 1, 1, 3)},   # Prior 1
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': black_box_function(2, 2, 2, 1)},   # Prior 2
            {'N1': 3, 'N2': 0, 'N3':0, 'N4':2, 'target': black_box_function(3, 0, 0, 2)},   # Prior 3
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': black_box_function(1, 3, 3, 0)},   #prior 4
            {'N1': 1, 'N2': 0, 'N3':2, 'N4':0, 'target': black_box_function(1, 0, 2, 0)},   #prior 5
        ]
    priors2 = [
            {'N1': 0, 'N2': 1, 'N3':1, 'N4':3, 'target': cost_function([0, 1, 1, 3])},   # Prior 1
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': cost_function([2, 2, 2, 1])},   # Prior 2
            {'N1': 3, 'N2': 0, 'N3':0, 'N4':2, 'target': cost_function([3, 0, 0, 2])},   # Prior 3
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': cost_function([1, 3, 3, 0])},   #prior 4
            {'N1': 1, 'N2': 0, 'N3':2, 'N4':0, 'target': cost_function([1, 0, 2, 0])},   #prior 5
        ]
    
    #3 
    priors = [ 
            {'N1': 3, 'N2': 3, 'N3':2, 'N4':1, 'target': black_box_function(3, 3, 2, 1)},   # Prior 1
            {'N1': 1, 'N2': 0, 'N3':0, 'N4':3, 'target': black_box_function(1, 0, 0, 3)},   # Prior 2
            {'N1': 0, 'N2': 2, 'N3':3, 'N4':0, 'target': black_box_function(0, 2, 3, 0)},   # Prior 3
            {'N1': 2, 'N2': 1, 'N3':1, 'N4':2, 'target': black_box_function(2, 1, 1, 2)},   #prior 4
            {'N1': 2, 'N2': 2, 'N3':0, 'N4':2, 'target': black_box_function(2, 2, 0, 2)},   #prior 5
        ]
    priors2 = [ 
            {'N1': 3, 'N2': 3, 'N3':2, 'N4':1, 'target': cost_function([3, 3, 2, 1])},   # Prior 1
            {'N1': 1, 'N2': 0, 'N3':0, 'N4':3, 'target': cost_function([1, 0, 0, 3])},   # Prior 2
            {'N1': 0, 'N2': 2, 'N3':3, 'N4':0, 'target': cost_function([0, 2, 3, 0])},   # Prior 3
            {'N1': 2, 'N2': 1, 'N3':1, 'N4':2, 'target': cost_function([2, 1, 1, 2])},   #prior 4
            {'N1': 2, 'N2': 2, 'N3':0, 'N4':2, 'target': cost_function([2, 2, 0, 2])},   #prior 5
        ]
    
    
    4
    priors = [
            {'N1': 1, 'N2': 3, 'N3':1, 'N4':2, 'target': black_box_function(1, 3, 1, 2)},   # Prior 1
            {'N1': 2, 'N2': 0, 'N3':2, 'N4':1, 'target': black_box_function(2, 0, 2, 1)},   # Prior 2
            {'N1': 3, 'N2': 2, 'N3':0, 'N4':3, 'target': black_box_function(3, 2, 0, 3)},   # Prior 3
            {'N1': 0, 'N2': 1, 'N3':3, 'N4':0, 'target': black_box_function(0, 1, 3, 0)},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':2, 'N4':0, 'target': black_box_function(0, 2, 2, 0)},   #prior 5
        ]

    priors2 = [
            {'N1': 1, 'N2': 3, 'N3':1, 'N4':2, 'target': cost_function([1, 3, 1, 2])},   # Prior 1
            {'N1': 2, 'N2': 0, 'N3':2, 'N4':1, 'target': cost_function([2, 0, 2, 1])},   # Prior 2
            {'N1': 3, 'N2': 2, 'N3':0, 'N4':3, 'target': cost_function([3, 2, 0, 3])},   # Prior 3
            {'N1': 0, 'N2': 1, 'N3':3, 'N4':0, 'target': cost_function([0, 1, 3, 0])},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':2, 'N4':0, 'target': cost_function([0, 2, 2, 0])},   #prior 5
        ]
    '''
    #5
    priors = [
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': black_box_function(1, 3, 3, 0)},   # Prior 1
            {'N1': 3, 'N2': 1, 'N3':0, 'N4':3, 'target': black_box_function(3, 1, 0, 3)},   # Prior 2
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': black_box_function(2, 2, 2, 1)},   # Prior 3
            {'N1': 0, 'N2': 0, 'N3':1, 'N4':2, 'target': black_box_function(0, 0, 1, 2)},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':0, 'N4':2, 'target': black_box_function(0, 2, 0, 2)},   #prior 5
        ]

    priors2 = [
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': cost_function([1, 3, 3, 0])},   # Prior 1
            {'N1': 3, 'N2': 1, 'N3':0, 'N4':3, 'target': cost_function([3, 1, 0, 3])},   # Prior 2
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': cost_function([2, 2, 2, 1])},   # Prior 3
            {'N1': 0, 'N2': 0, 'N3':1, 'N4':2, 'target': cost_function([0, 0, 1, 2])},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':0, 'N4':2, 'target': cost_function([0, 2, 0, 2])},   #prior 5
        ]
    '''
    '''
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
        Z = np.array([float(p) for p in Z])

        # Fit the Gaussian process regressor
        regressor.fit(P, Z)

        # Initialize the Gaussian process regressor 2
        regressor2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                              normalize_y=True,
                                              n_restarts_optimizer=5,
                                              random_state=13)

        # Prepare the data for Gaussian process regression
        P2 = np.array([[p['N1'], p['N2'], p['N3'], p['N4']] for p in priors2])
        Z2 = np.array([p['target'] for p in priors2])

        # Fit the Gaussian process regressor
        regressor2.fit(P2, Z2)

        problem = CostProblem(regressor, regressor2, kappa)

        # Define the NSGA-II algorithm
        algorithm = NSGA2(pop_size=50, sampling=IntegerRandomSampling(),
                          crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          eliminate_duplicates=True)

        # Find pareto frontier
        res = minimize(problem, algorithm, termination=('n_gen', 4), verbose=False)

        # Select best point based on uncertainty
        measure = []
        for solution in res.X:
            # Predict the mean and standard deviation for each objective using both models
            mu1, sigma1 = regressor.predict([solution], return_std=True)
            mu2, sigma2 = regressor2.predict([solution], return_std=True)

            # Compute upper confidence bound (UCB) and lower confidence bound (LCB) for both objectives
            ucb1 = mu1 + kappa * sigma1
            lcb1 = mu1 - kappa * sigma1
            ucb2 = mu2 + kappa * sigma2
            lcb2 = mu2 - kappa * sigma2

            # Calculate the difference between UCB and LCB for each objective
            delta_ucb1 = ucb1[0] - lcb1[0]
            delta_ucb2 = ucb2[0] - lcb2[0]

            # Calculate the volume of the uncertainty hyper-rectangle
            uncertainty_volume = delta_ucb1 * delta_ucb2
            measure.append(uncertainty_volume)

        measure = np.array(measure)
        # Find the index of the solution with the maximum uncertainty volume
        best_index = np.argmax(measure)
        # Retrieve the best solution and its corresponding objective values
        best_solution = res.X[best_index]
        best_objectives = res.F[best_index]

        # Evaluate the black-box function for the best_solution
        if np.array_equal(best_solution, [0, 0, 0, 0]):
            best_prior = {'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'target': worst_performance}
            best_performance = worst_performance
        else:
            # Append the best_solution and its performance to the list of priors
            best_prior, best_performance = call_initializer(best_solution)
        best_cost = cost_function(list(best_solution))

        # Append the best_solution and its performance to the list of priors
        best_prior = {
            'N1': int(best_solution[0]),
            'N2': int(best_solution[1]),
            'N3': int(best_solution[2]),
            'N4': int(best_solution[3]),
            'target': best_performance
        }
        best_prior2 = {
            'N1': int(best_solution[0]),
            'N2': int(best_solution[1]),
            'N3': int(best_solution[2]),
            'N4': int(best_solution[3]),
            'target': best_cost
        }

        priors.append(best_prior)
        priors2.append(best_prior2)
        print(f"Point suggestion: {str(best_solution)}, value: {str(float(best_performance) + best_cost)}")
        count += 1

    visited_performance = np.array([float(p['target']) for p in priors])
    visited_crews = np.array([[p['N1'], p['N2'], p['N3'], p['N4']] for p in priors])
    visited_cost = np.array([cost_function(list(visited_crew)) for visited_crew in visited_crews])

    print("visited_crews", visited_crews)
    print('visited_performance', visited_performance)
    print("visited_cost", visited_cost)

    # Convert visited_crews array to a list of strings to use as x-axis ticks
    x_data = [' '.join(map(str, crew)) for crew in visited_crews]

    # Create an array of indices for x-axis positioning
    x_indices = np.arange(len(x_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x_indices, visited_performance, visited_cost, c='r', marker='o')

    # Setting x-ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_data, rotation=45, fontsize = 8)  # Rotate for readability

    ax.set_xlabel('Crew Combinations')
    ax.set_ylabel('Total Time')
    ax.set_zlabel('Cost')
    plt.ylim(0,2000);
    plt.tight_layout()
    plt.title("USeMO Result")
    visited_utility = visited_performance + visited_cost
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
    plt.bar(x_indices, visited_performance, color='blue')
    plt.xticks(x_indices, x_data, rotation=45, fontsize=8)
    plt.ylabel('Total Time')
    plt.xlabel('Crew Combinations')
    plt.tight_layout()
    plt.title("UseMO Total Time")
    plt.ylim(0, 2000);
    plt.show()

    # Second 2D graph: Crew Combinations vs Cost
    plt.figure(figsize=(10, 5))
    plt.bar(x_indices, visited_cost, color='green')
    plt.xticks(x_indices, x_data, rotation=45, fontsize=8)
    plt.ylabel('Cost')
    plt.xlabel('Crew Combinations')
    plt.title("UseMO Cost")
    plt.tight_layout()
    plt.show()