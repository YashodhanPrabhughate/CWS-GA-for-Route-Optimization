import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
# Replace 'multiple_runs_data.csv' with the path to your actual data file
# The file should contain columns: 'Run', 'ACO_Best_Distance', 'Hybrid_Best_Distance'
data = pd.read_csv("convergence_data.csv")

# Calculate the standard deviation of the best route distances across multiple runs
std_aco = np.std(data['Aco'])
std_hybrid = np.std(data['Hybrid'])

# Print out the standard deviations for both algorithms
print(f"Standard Deviation of Best Route Distance for ACO: {std_aco:.4f}")
print(f"Standard Deviation of Best Route Distance for Hybrid: {std_hybrid:.4f}")

# Create a bar plot comparing the standard deviations
plt.figure(figsize=(8, 5))
algorithms = ['ACO', 'Hybrid']
std_values = [std_aco, std_hybrid]

plt.bar(algorithms, std_values, color=['lightblue', 'lightgreen'])
plt.title('Stability Comparison: Standard Deviation of Best Route Distance Across Multiple Runs')
plt.ylabel('Standard Deviation of Best Route Distance')
plt.xlabel('Algorithm')
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
