import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load data (no change needed)
def load_record_data():
    data = pd.read_csv('locations_dataset.csv')
    return data[['x', 'y']].to_numpy()

# Define constants
ALPHA = 1  # Pheromone influence
BETA = 2   # Heuristic influence
EVAPORATION_RATE = 0.5
Q = 100     # Total amount of pheromone
NUM_ANTS = 50
NUM_GENERATIONS = 100

# ACO Class
class ACO:
    def __init__(self, locations):
        self.locations = locations
        self.num_bins = len(locations)
        self.distance_matrix = self.compute_distance_matrix()
        self.pheromone = np.ones((self.num_bins, self.num_bins))  # Initialize pheromone levels

    def compute_distance_matrix(self):
        # Euclidean distance between bins
        return np.array([[np.linalg.norm(self.locations[i] - self.locations[j]) for j in range(self.num_bins)] for i in range(self.num_bins)])

    def select_next_bin(self, current_bin, visited):
        probabilities = np.zeros(self.num_bins)
        epsilon = 1e-10  # A small constant to avoid division by zero
        for i in range(self.num_bins):
            if i not in visited:
                # Calculate heuristic for unvisited bins
                heuristic = (1 / (self.distance_matrix[current_bin][i] + epsilon)) ** BETA
                pheromone = self.pheromone[current_bin][i] ** ALPHA
                probabilities[i] = pheromone * heuristic

        total = np.sum(probabilities)
        if total == 0:
            # Assign equal probability to all unvisited bins
            probabilities = np.ones(self.num_bins) * (1 / (self.num_bins - len(visited)))
        else:
            probabilities /= total

        probabilities = np.nan_to_num(probabilities)  # Replace NaNs with zeros
        return np.random.choice(range(self.num_bins), p=probabilities)

    def construct_solution(self):
        path = [0]  # Start from the depot
        visited = set(path)
        while len(visited) < self.num_bins:
            current_bin = path[-1]
            next_bin = self.select_next_bin(current_bin, visited)
            path.append(next_bin)
            visited.add(next_bin)
        return path

    def update_pheromone(self, all_paths, all_distances):
        # Evaporate pheromone
        self.pheromone *= (1 - EVAPORATION_RATE)
        # Deposit pheromone
        for path, distance in zip(all_paths, all_distances):
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += Q / distance

    def run(self):
        best_path = None
        best_distance = float('inf')
        best_distances = []
        diversity_data = []

        for generation in range(NUM_GENERATIONS):
            all_paths = []
            all_distances = []
            for ant in range(NUM_ANTS):
                path = self.construct_solution()
                distance = self.calculate_path_distance(path)
                all_paths.append(path)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_path = path

            # Update pheromone levels
            self.update_pheromone(all_paths, all_distances)

            best_distances.append(best_distance)

            diversity = self.calculate_diversity(all_paths)
            diversity_data.append({'Generation': generation, 'Diversity': diversity, 'Best_Distance': best_distance})
        
            print(f"Generation {generation}: Best distance = {best_distance}")

        diversity_df = pd.DataFrame(diversity_data)
        diversity_df.to_csv("aco_diversity_data.csv", index=False)

        return best_path, best_distance, best_distances

    def calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i], path[i+1]]
        distance += self.distance_matrix[path[-1], 0]  # Return to depot
        return distance
    
    def calculate_diversity(self, all_paths):
        diversity = 0
        n_ants = len(all_paths)
        
        # Calculate pairwise distance between ants' paths
        for i in range(n_ants):
            for j in range(i + 1, n_ants):
                distance = 0
                # Calculate Euclidean distance between the two routes
                for k in range(len(all_paths[i])):
                    distance += np.linalg.norm(self.locations[all_paths[i][k]] - self.locations[all_paths[j][k]])
                diversity += distance
        
        # Normalize diversity by number of comparisons
        return diversity / (n_ants * (n_ants - 1) / 2)

# Visualization
def plot_routes(locations, path):
    plt.figure(figsize=(8, 6))
    for i in range(len(path) - 1):
        plt.plot([locations[path[i], 0], locations[path[i + 1], 0]], 
                 [locations[path[i], 1], locations[path[i + 1], 1]], 'bo-')
    plt.plot([locations[path[-1], 0], locations[path[0], 0]], 
             [locations[path[-1], 1], locations[path[0], 1]], 'bo-')
    plt.scatter(locations[:, 0], locations[:, 1], color='red')
    for i, location in enumerate(locations):
        plt.text(location[0], location[1], str(i), fontsize=12)
    plt.title("Best Route Found by ACO")
    plt.show()

# Main ACO execution
def run_aco():
    locations = load_record_data()  # Load all data from CSV
    aco = ACO(locations)
    best_path, best_distance, best_distances = aco.run()
    print("Best Route:", best_path)
    plot_routes(locations, best_path)
    return best_distances

# Run the ACO algorithm
aco_distances = run_aco()
