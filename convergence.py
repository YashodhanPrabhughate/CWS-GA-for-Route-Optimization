import pandas as pd
import matplotlib.pyplot as plt

# Load convergence data from CSV
convergence_df = pd.read_csv("convergence_data.csv")

# Plot comparison graph
plt.figure(figsize=(10, 6))
plt.plot(convergence_df['Generation'], convergence_df['Hybrid'], marker='o', color='b', label='Hybrid Algorithm')
plt.plot(convergence_df['Generation'], convergence_df['Aco'], marker='s', color='r', label='ACO')
plt.xlabel('Generations')
plt.ylabel('Best Route Distance')
plt.title('Convergence Comparison: Hybrid Algorithm vs ACO')
plt.legend()
plt.grid()
plt.show()
