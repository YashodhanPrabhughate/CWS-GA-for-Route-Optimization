import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools

 # Load locations from CSV
locations_df = pd.read_csv("locations_dataset.csv")
locations = locations_df[['x', 'y']].values  # Convert DataFrame to NumPy array
n_bins = len(locations)
depot = 0  # Depot location index

 # Calculate Euclidean distance between two points
def euclidean_distance(loc1, loc2):
     return np.linalg.norm(loc1 - loc2)

 # Distance matrix
distance_matrix = np.array([[euclidean_distance(locations[i], locations[j]) for j in range(n_bins)] for i in range(n_bins)])


# Clarke-Wright Savings Algorithm
def clarke_wright_savings():
    savings = []
    for i in range(1, n_bins):
        for j in range(i + 1, n_bins):
            savings_value = distance_matrix[depot, i] + distance_matrix[depot, j] - distance_matrix[i, j]
            savings.append((savings_value, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    routes = [[i] for i in range(1, n_bins)]
    for s, i, j in savings:
        route_i = next(route for route in routes if i in route)
        route_j = next(route for route in routes if j in route)
        if route_i != route_j:
            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(route_i + route_j)

    return routes

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    routes = clarke_wright_savings()
    return creator.Individual(routes)

def evaluate_individual(individual):
    total_distance = 0
    for route in individual:
        total_distance += distance_matrix[depot, route[0]]
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i], route[i + 1]]
        total_distance += distance_matrix[route[-1], depot]
    return (total_distance,)

def crossover(parent1, parent2):
    child1, child2 = parent1[:], parent2[:]
    if len(parent1) > 1 and len(parent2) > 1:
        cut_point = random.randint(1, len(parent1) - 1)
        child1[:cut_point], child2[:cut_point] = parent2[:cut_point], parent1[:cut_point]
    return child1, child2

def mutate(individual):
    if len(individual) > 1:
        route1, route2 = random.sample(range(len(individual)), 2)
        idx1, idx2 = random.randint(0, len(individual[route1])-1), random.randint(0, len(individual[route2])-1)
        individual[route1][idx1], individual[route2][idx2] = individual[route2][idx2], individual[route1][idx1]
    return individual,

# 2-opt Local Search Improvement
def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_route_distance(new_route) < calculate_route_distance(best):
                    best = new_route
                    improved = True
    return best

def calculate_route_distance(route):
    distance = distance_matrix[depot, route[0]]
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i], route[i + 1]]
    distance += distance_matrix[route[-1], depot]
    return distance

# Apply 2-opt local search after genetic evolution
def local_search(individual):
    for i, route in enumerate(individual):
        individual[i] = two_opt(route)
    return individual

# def calculate_diversity(population):
#     route_distances = [calculate_route_distance(individual) for individual in population]
#     diversity = np.var(route_distances)  # Variance is one measure of diversity
#     return diversity

# Main Genetic Algorithm execution
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("local_search", local_search)

def genetic_algorithm(pop_size=100, cx_prob=0.8, mut_prob=0.4, n_gen=100):
    population = toolbox.population(n=pop_size)
    # convergence_data = []

    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # diversity_data = []

    for gen in range(n_gen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Apply 2-opt local search
        for ind in population:
            toolbox.local_search(ind)
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring

        # diversity = calculate_diversity(population)
        best_distance = min(ind.fitness.values[0] for ind in population)
        # diversity_data.append({
        #     'Generation': gen,
        #     'Diversity': diversity,
        #     'Best_Distance': min(fitnesses)[0]  # Store best distance for tracking progress
        # })

        # Gather stats
        fits = [ind.fitness.values[0] for ind in population]
        best_ind = tools.selBest(population, 1)[0]
        # best_distance = best_ind.fitness.values[0]
        # convergence_data.append(best_distance)
        print(f"Generation {gen}: Best route distance {best_ind.fitness.values[0]}")

        # convergence_df = pd.DataFrame({'Generation': list(range(len(convergence_data))), 'Best_Distance': convergence_data})
        # convergence_df.to_csv("convergence_data.csv", index=False)

        # diversity_df = pd.DataFrame(diversity_data)
        # diversity_df.to_csv("diversity_data.csv", index=False)

    return best_ind

# Visualization function
def plot_routes(individual):
    G = nx.Graph()
    for route in individual:
        for i in range(len(route) - 1):
            G.add_edge(route[i], route[i + 1], weight=distance_matrix[route[i], route[i + 1]])
        # Connect the route's last location back to the depot
        G.add_edge(route[-1], depot, weight=distance_matrix[route[-1], depot])
        # Connect the depot to the route's first location
        G.add_edge(depot, route[0], weight=distance_matrix[depot, route[0]])

    # Define the position of each node based on `locations`
    pos = {i: locations[i] for i in range(len(locations))}
    pos[depot] = locations[depot] 
    
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='blue', style='-')
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Label each node with its index
    for i, (x, y) in pos.items():
        plt.text(x, y, str(i), fontsize=12, ha='right', va='top', color='black')

    plt.title("Best Route Found by CWS-GA Algorithm")
    plt.show()

# Running the Genetic Algorithm
best_route = genetic_algorithm()
print("Best route:", best_route)
plot_routes(best_route)
