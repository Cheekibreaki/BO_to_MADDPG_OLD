# BENCHMARK 2
# install pymoo 0.6.0.1

import json
import os
import numpy as np
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
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
import subprocess

np.random.seed(2609)
random.seed(2609)
# interpreter_path = "C:/Users/Daniel Yin/AppData/Local/Programs/Python/Python39/python.exe"
interpreter_path = "E:/Summer Research 2023/DME-DRL Daniel/DME_DRL_CO/venv/Scripts/python.exe "
best_crew_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/BOOF_best_crew.json "
base_config_path = "E:/Summer Research 2023/BO_to_MADDPG/BO_to_MADDPG/base_config_map4_1.yaml "
test_run_config_path = "E:/Summer Research 2023/MADDPG_New/MADDPG/assets/test_run_config.yaml"


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None, dtype=object)

        # Create grid of discrete points in the search space
        N_space = [0, 1, 2, 3]
        grid_points = np.array(list(product(N_space, repeat=problem.n_var)))

        # Predict LCB scores for all grid points
        mu, sigma = problem.regressor.predict(grid_points, return_std=True)
        LCB_scores = mu - problem.kappa * sigma

        # Find the indices of the n_samples points with the lowest LCB scores
        min_LCB_indices = np.argsort(LCB_scores)[:n_samples]

        # Extract the best n_samples points from the grid points
        selected_samples = grid_points[min_LCB_indices]

        # Fill the population matrix with the selected samples
        X[:n_samples, :] = selected_samples

        return X


def cost_function(q):
    # Fixed costs for each feature level
    cost_low_low = 50
    cost_high_low = 100
    cost_low_high = 75
    cost_high_high = 150

    if isinstance(q, list):
        total_cost = q[0] * cost_low_low + q[1] * cost_high_low + q[2] * cost_low_high + q[3] * cost_high_high

    else:
        # Calculate the total cost of the crew
        total_cost = q[:, 0] * cost_low_low + q[:, 1] * cost_high_low + q[:, 2] * cost_low_high + q[:,
                                                                                                  3] * cost_high_high

    return total_cost


def get_random_satisfaction(crew):
    # Check if the satisfaction for the crew is already in the dictionary
    if tuple(crew) in satisfaction_dict:
        return satisfaction_dict[tuple(crew)]
    else:
        # Generate a random satisfaction value between 0 and 100
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


# define the black box function
def black_box_function(q_low_low, q_high_low, q_low_high, q_high_high):
    # Get the random satisfaction value for the crew
    satisfaction = get_random_satisfaction([q_low_low, q_high_low, q_low_high, q_high_high])

    # if q_low_low==0 and q_high_low==0 and q_low_high==0 and q_high_high ==0:

    #   satisfaction = 0
    #   print('0 VALUES, satisfaction', satisfaction)
    return -1 * satisfaction


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
    def __init__(self, regressor, kappa):
        super().__init__(n_var=4, n_obj=2, n_constr=0, xl=np.array([0, 0, 0, 0]), xu=np.array([3, 3, 3, 3]))
        self.regressor = regressor
        self.kappa = kappa

    def _evaluate(self, X, out, *args, **kwargs):
        # to do: call performance from f1
        mu, sigma = self.regressor.predict(X, return_std=True)
        # LCB score as performance
        f1 = mu - self.kappa * sigma
        # Cost function:
        f2 = cost_function(X)

        out["F"] = np.column_stack([f1, f2])


# Debugging purpose
class MyCallback(Callback):

    def notify(self, algorithm):
        # Get the current generation number
        gen = algorithm.n_gen

        if gen == 1:
            # Print the initial population generated by Pymoo
            initial_population = algorithm.pop.get("X")
            print(f"Initial Population (Generation {gen}):\n{initial_population}\n")
        else:
            # Get the current population
            pop = algorithm.pop.get("X")

            # Print the current population
            print(f"Population (Generation {gen}):\n{pop}\n")

        # Problem Hiperparameters


# subprocess call
def call_initializaer(solution):
    # Specify the file path
    file_path = os.getcwd() + '/BOOF_best_crew.json'

    crew = np.array(solution)
    # Write the data to the JSON file
    with open(file_path, 'w') as file:
        json.dump(crew.tolist(), file)
    try:
        result = subprocess.run([interpreter_path, 'exploration_initializer.py',
                                 file_path, base_config_path, base_config_path], check=True, cwd=os.getcwd(),
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


budget = 10
satisfaction_dict = {}  # Dictionary to store satisfaction values for each unique crew
satisfaction_lower_bound = 0
satisfaction_upper_bound = 750  # 150 * 5

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

    # Initial Priors
    # 1
    priors = [
        {'N1': 2, 'N2': 0, 'N3': 0, 'N4': 3, 'target': black_box_function(2, 0, 0, 3)},  # Prior 1
        {'N1': 0, 'N2': 3, 'N3': 3, 'N4': 0, 'target': black_box_function(0, 3, 3, 0)},  # Prior 2
        {'N1': 1, 'N2': 1, 'N3': 1, 'N4': 2, 'target': black_box_function(1, 1, 1, 2)},  # Prior 3
        {'N1': 3, 'N2': 2, 'N3': 2, 'N4': 1, 'target': black_box_function(3, 2, 2, 1)},  # prior 4
        {'N1': 3, 'N2': 1, 'N3': 3, 'N4': 1, 'target': black_box_function(3, 1, 3, 1)},  # prior 5
    ]

    '''
    #2
    priors = [ 
            {'N1': 0, 'N2': 1, 'N3':1, 'N4':3, 'target': black_box_function(0, 1, 1, 3)},   # Prior 1
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': black_box_function(2, 2, 2, 1)},   # Prior 2
            {'N1': 3, 'N2': 0, 'N3':0, 'N4':2, 'target': black_box_function(3, 0, 0, 2)},   # Prior 3
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': black_box_function(1, 3, 3, 0)},   #prior 4
            {'N1': 1, 'N2': 0, 'N3':2, 'N4':0, 'target': black_box_function(1, 0, 2, 0)},   #prior 5
        ]


    #3 
    priors = [ 
            {'N1': 3, 'N2': 3, 'N3':2, 'N4':1, 'target': black_box_function(3, 3, 2, 1)},   # Prior 1
            {'N1': 1, 'N2': 0, 'N3':0, 'N4':3, 'target': black_box_function(1, 0, 0, 3)},   # Prior 2
            {'N1': 0, 'N2': 2, 'N3':3, 'N4':0, 'target': black_box_function(0, 2, 3, 0)},   # Prior 3
            {'N1': 2, 'N2': 1, 'N3':1, 'N4':2, 'target': black_box_function(2, 1, 1, 2)},   #prior 4
            {'N1': 2, 'N2': 2, 'N3':0, 'N4':2, 'target': black_box_function(2, 2, 0, 2)},   #prior 5
        ]

    #4 
    priors = [ 
            {'N1': 1, 'N2': 3, 'N3':1, 'N4':2, 'target': black_box_function(1, 3, 1, 2)},   # Prior 1
            {'N1': 2, 'N2': 0, 'N3':2, 'N4':1, 'target': black_box_function(2, 0, 2, 1)},   # Prior 2
            {'N1': 3, 'N2': 2, 'N3':0, 'N4':3, 'target': black_box_function(3, 2, 0, 3)},   # Prior 3
            {'N1': 0, 'N2': 1, 'N3':3, 'N4':0, 'target': black_box_function(0, 1, 3, 0)},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':2, 'N4':0, 'target': black_box_function(0, 2, 2, 0)},   #prior 5
        ]    


    #5 
    priors = [ 
            {'N1': 1, 'N2': 3, 'N3':3, 'N4':0, 'target': black_box_function(1, 3, 3, 0)},   # Prior 1
            {'N1': 3, 'N2': 1, 'N3':0, 'N4':3, 'target': black_box_function(3, 1, 0, 3)},   # Prior 2
            {'N1': 2, 'N2': 2, 'N3':2, 'N4':1, 'target': black_box_function(2, 2, 2, 1)},   # Prior 3
            {'N1': 0, 'N2': 0, 'N3':1, 'N4':2, 'target': black_box_function(0, 0, 1, 2)},   #prior 4
            {'N1': 0, 'N2': 2, 'N3':0, 'N4':2, 'target': black_box_function(0, 2, 0, 2)},   #prior 5
        ]
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

        # Fit the Gaussian process regressor
        regressor.fit(P, Z)

        problem = CostProblem(regressor, kappa)
        '''
        #Build bias pop f(n)=LCB(n) - C(n)
        mu0, sigma0 = regressor.predict(grid_points, return_std=True)
        #LCB score as performance
        p0 = mu0 - kappa * sigma0
        #Cost function:
        c0 = cost_function(grid_points)
        #Approximate performance
        f0 = p0+c0 

        # Find the indices of the 100 points with the lowest f(n) values
        min_f0_indices = np.argsort(f0)[:50]

        # Extract the best 100 points from the grid points
        bias_ini_pop = grid_points[min_f0_indices]
        '''

        '''
        #Define the NSGA-II algorithm 
        algorithm = NSGA2(pop_size=5, sampling=IntegerRandomSampling(),             
                        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                        eliminate_duplicates=True)
        '''
        # Define the BIAS NSGA-II algorithm
        algorithm = NSGA2(pop_size=50, sampling=MySampling(),
                          crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          eliminate_duplicates=True)

        # Create an instance of your custom callback

        # my_callback = MyCallback()
        # print('bias initial pop',bias_ini_pop)

        # Find pareto frontier
        res = minimize(problem, algorithm=algorithm, termination=('n_gen', 4), verbose=False)

        # print('Pareto frontier',res.X, res.F )
        # Select best point based on uncertainty
        measure = res.F[:, 0]  # + res.F[:,1]
        # Find the index of the solution with the max std
        best_index = np.argmin(measure)

        # Retrieve the best solution and its corresponding objective values
        best_solution = res.X[best_index]
        best_objectives = res.F[best_index]

        # Evaluate the black-box function for the best_solution
        # TODO: require subprocess here for the best crew
        # best_performance = black_box_function(best_solution[0], best_solution[1], best_solution[2], best_solution[3])
        best_cost = cost_function(list(best_solution))

        # Append the best_solution and its performance to the list of priors
        best_prior, best_performance = call_initializaer(best_solution)
        # best_prior = {
        #     'N1': int(best_solution[0]),
        #     'N2': int(best_solution[1]),
        #     'N3': int(best_solution[2]),
        #     'N4': int(best_solution[3]),
        #     'target': best_performance
        # }

        # if best_prior in priors:
        #    print('repeated prior')
        #    break
        # else:
        priors.append(best_prior)
        print("Point suggestion : {}, value: {}".format(best_solution, best_performance + best_cost))
        count += 1

    visited_performance = np.array([p['target'] for p in priors])
    visited_crews = np.array([[p['N1'], p['N2'], p['N3'], p['N4']] for p in priors])
    visited_utility = visited_performance + cost_function(visited_crews)
    best_visited_utility = np.argmin(visited_utility)
    best_crew = visited_crews[best_visited_utility]
    print("Best point suggestion : {}, value: {}".format(best_crew, np.min(visited_utility)))
    max_utility, max_crew = max_satisfaction()
    print("The best crew is [{}, {}, {}, {}], the max value is {}".format(max_crew[0], max_crew[1], max_crew[2],
                                                                          max_crew[3], max_utility))
    # print("best crew:", best_crew)

    # # Specify the file path
    # file_path = os.getcwd() + '/BOOF_best_crew.json'
    #
    # best_crew = np.array(best_crew)
    # # Write the data to the JSON file
    # result = ""
    # with open(file_path, 'w') as file:
    #     json.dump(best_crew.tolist(), file)
    # try:
    #     result = subprocess.run([interpreter_path, 'exploration_initializer.py',
    #                              best_crew_path , base_config_path , test_run_config_path], check=True, cwd=os.getcwd(),
    #                             stdout=subprocess.PIPE, text=True, encoding='utf-8')
    #     result = result.stdout.splitlines()[-1]  # The standard output of the subprocess
    #     # Now 'result' is properly defined within the try block
    # except subprocess.CalledProcessError as e:
    #     print(f"Error running exploration script_path: {e}")
    #     raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # new_data = {
    #     'N1': max_crew[0],
    #     'N2': max_crew[1],
    #     'N3': max_crew[2],
    #     'N4': max_crew[3],
    #     'target': result  # Adjust the arguments as necessary
    # }
