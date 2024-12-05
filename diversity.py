import pandas as pd
import matplotlib.pyplot as plt

# Load diversity data for ACO and Hybrid algorithms from CSV files
def load_diversity_data(aco_file, hybrid_file):
    # Load the ACO and Hybrid algorithm diversity data
    aco_data = pd.read_csv(aco_file)
    hybrid_data = pd.read_csv(hybrid_file)
    
    return aco_data, hybrid_data

# Calculate and plot diversity comparison
def plot_diversity_comparison(aco_data, hybrid_data):
    # Assuming the CSV files have columns: 'Generation' and 'Diversity'
    aco_diversity = aco_data['Diversity']
    hybrid_diversity = hybrid_data['Diversity']
    
    generations_aco = aco_data['Generation']
    generations_hybrid = hybrid_data['Generation']
    
    # Ensure both datasets have the same number of generations for comparison
    if len(generations_aco) != len(generations_hybrid):
        min_len = min(len(generations_aco), len(generations_hybrid))
        generations_aco = generations_aco[:min_len]
        generations_hybrid = generations_hybrid[:min_len]
        aco_diversity = aco_diversity[:min_len]
        hybrid_diversity = hybrid_diversity[:min_len]

    # Plotting the diversity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(generations_aco, aco_diversity, label="ACO", color='blue')
    plt.plot(generations_hybrid, hybrid_diversity, label="Hybrid Algorithm", color='green')
    plt.xlabel("Generations")
    plt.ylabel("Route Distance Variance (Diversity)")
    plt.title("Diversity of Solutions Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
def main():
    # Load ACO and Hybrid algorithm diversity data from CSV files
    aco_file = "aco_diversity_data.csv" 
    hybrid_file = "diversity_data.csv"  
    
    aco_data, hybrid_data = load_diversity_data(aco_file, hybrid_file)
    
    # Calculate and plot the diversity comparison
    plot_diversity_comparison(aco_data, hybrid_data)

# Run the main function
if __name__ == "__main__":
    main()
