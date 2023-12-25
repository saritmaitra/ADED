# -*- coding: utf-8 -*-
"""ADED.py



## ADED Algorithm for convext and non convex objective function
"""

def convex_function(x):
    if isinstance(x, (int, float)):
        """If x is a single value, convert it to a list"""
        x = [x]
    return np.sum(np.fromiter((xi**2 for xi in x), dtype=float))

def sinusoidal_function(x):
    if isinstance(x, (int, float)):
        return np.sin(x)
    elif isinstance(x, np.ndarray):
        return np.sin(x[0]) + np.cos(x[1])
    else:
        raise ValueError("Unsupported input type for sinusoidal_function")

"""ADED Algorithm Implementation"""
def ADED(objective_function, bounds, population_size, max_generations):
    def initialize_population(population_size, bounds):
        return [np.random.uniform(low, high, len(bounds)) for _ in range(population_size)]

    def initialize_parameters():
        return np.random.uniform(0.5, 2.0), np.random.uniform(0.1, 0.9)

    def adapt_parameters(F, CR, fitness, avg_fitness):
        """Simple adaptation: F and CR remain constant"""
        return F, CR

    def dynamic_neighborhood(individual_index, neighborhoods):
        """Dynamic neighborhood topology (randomly select neighbors)"""
        all_individuals = list(range(len(neighborhoods)))
        all_individuals.remove(individual_index)  # Exclude self from potential neighbors
        neighborhood_size = min(len(all_individuals), len(neighborhoods[individual_index]))
        return np.random.choice(all_individuals, neighborhood_size, replace=False)

    def update_neighborhoods(individual_index, neighbors, trial_fitness, neighborhoods):
        """Update neighborhood based on trial fitness"""
        for neighbor_index in neighbors:
            if trial_fitness < evaluate_fitness(neighborhoods[individual_index][neighbor_index]):
                neighborhoods[individual_index][neighbor_index] = trial_fitness

    def generate_trial_solution(individual, population, F, CR):
        r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
        r1, r2, r3 = population[r1], population[r2], population[r3]
        return individual + F * (r1 - individual) + F * (r2 - r3)

    def evaluate_fitness(individual):
        return objective_function(individual)

    def average_fitness(population):
        return np.mean([objective_function(individual) for individual in population])

    def crowding_selection(individual1, individual2, neighbors):
        """Select the individual with better fitness"""
        if evaluate_fitness(individual1) < evaluate_fitness(individual2):
            return individual1
        else:
            return individual2

    """Initialization"""
    dimension = len(bounds)
    low, high = zip(*bounds)  """Unzip bounds into 'low' and 'high' lists"""
    population = initialize_population(population_size, bounds)
    F, CR = initialize_parameters()
    neighborhoods = [list(range(population_size)) for _ in range(population_size)]
    best_solutions = [None] * population_size

    best_fitness_history = []

    for generation in range(max_generations):
        new_population = []

        for i, individual in enumerate(population):
            F, CR = adapt_parameters(F, CR, evaluate_fitness(individual), average_fitness(population))
            neighbors = dynamic_neighborhood(i, neighborhoods)
            trial_solution = generate_trial_solution(individual, population, F, CR)
            trial_fitness = evaluate_fitness(trial_solution)
            selected_individual = crowding_selection(individual, trial_solution, neighbors)
            update_neighborhoods(i, neighbors, trial_fitness, neighborhoods)
            new_population.append(selected_individual)

            if best_solutions[i] is None or trial_fitness < evaluate_fitness(best_solutions[i]):
                best_solutions[i] = trial_solution

        population = new_population

        """Record the best fitness for each individual in this generation"""
        best_fitness_history.append([evaluate_fitness(ind) for ind in best_solutions])

    return best_solutions, best_fitness_history

"""## ADED for multi-objective objective function"""

import numpy as np
from scipy.optimize import minimize

"""Multi_objective_function"""
def multi_objective_function(x):
    x_array = np.atleast_1d(x)

    if len(x_array) < 2:
        raise ValueError("Invalid input dimension. The input should have at least two elements.")

    obj1 = np.sin(x_array[0]) + np.cos(x_array[1])
    obj2 = np.exp(-(x_array[0] - 5)**2 - (x_array[1] - 5)**2)
    return np.array([obj1, obj2])

"""Scalarization function for multi-objective optimization"""
def scalarization_function(objective_values):

    weights = np.array([0.5, 0.5])
    return np.dot(objective_values, weights)

"""Pareto dominance check function"""
def pareto_dominance(obj_values1, obj_values2):
    return all(val1 <= val2 for val1, val2 in zip(obj_values1, obj_values2)) and any(val1 < val2 for val1, val2 in zip(obj_values1, obj_values2))

"""ADED Multi-Objective Algorithm"""
def ADED_multi_objective(objective_function, bounds, population_size, max_generations, stagnation_limit=10,
                           initial_mutation_rate=0.5, initial_crossover_rate=0.9):
    def initialize_population(population_size, bounds):
        return [np.random.uniform(low, high, len(bounds)) for _ in range(population_size)]

    def adaptive_mutation_rate(generation, max_generations):
        """Adjust mutation rate based on generation progress"""
        return initial_mutation_rate * (1.0 - generation / max_generations)

    def adaptive_crossover_rate(generation, max_generations):
        """Adjust crossover rate based on generation progress"""
        return initial_crossover_rate * (generation / max_generations)

    def has_converged(best_fitness_history, stagnation_limit):
        if len(best_fitness_history) < stagnation_limit:
            return False
        return np.array_equal(best_fitness_history[-stagnation_limit:], best_fitness_history[-1])


    def local_search(solution):
        result = minimize(lambda x: scalarization_function(objective_function(x)), solution, method='L-BFGS-B', bounds=bounds)
        return result.x

    def pareto_dominance(obj_values1, obj_values2):
        return all(val1 <= val2 for val1, val2 in zip(obj_values1, obj_values2)) and any(val1 < val2 for val1, val2 in zip(obj_values1, obj_values2))

    dimension = len(bounds)
    low, high = zip(*bounds)
    population = initialize_population(population_size, bounds)
    F, CR = adaptive_mutation_rate(0, max_generations), adaptive_crossover_rate(0, max_generations)
    neighborhoods = [list(range(len(population))) for _ in range(len(population))]
    best_solution = None
    best_fitness_history = []

    for generation in range(max_generations):
        new_population = []

       """ debugging"""
        # print(f"Generation {generation + 1}:")
        # print("Best Solution:", best_solution)

        """Update mutation and crossover rates"""
        F, CR = adaptive_mutation_rate(generation, max_generations), adaptive_crossover_rate(generation, max_generations)

        for i, individual in enumerate(population):
            neighbors = list(range(len(population)))
            trial_solution = individual + F * (population[np.random.choice(neighbors)] - individual) + F * (population[np.random.choice(neighbors)] - population[np.random.choice(neighbors)])

            """Apply local search to the trial solution"""
            trial_solution = local_search(trial_solution)
            trial_fitness = objective_function(trial_solution)

            """Pareto dominance check"""
            dominated = any(pareto_dominance(trial_fitness, objective_function(ind)) for ind in population)
            if not dominated:
                new_population.append(trial_solution)

            if best_solution is None or pareto_dominance(trial_fitness, objective_function(best_solution)):
                best_solution = trial_solution

        """Update population outside the loop"""
        population = new_population

        """Convergence check based on best solutions"""
        best_fitness_history.append(objective_function(best_solution))
        if has_converged(best_fitness_history, stagnation_limit):
            break

    return best_solution