import numpy as np
import matplotlib.pyplot as plt

# Objective function (minimize this)
def objective_function(x):
    return x**2 + 2*x + 1

# Simulated Annealing Algorithm
def simulated_annealing(initial_solution, objective_function, temperature, cooling_rate, iterations):
    current_solution = initial_solution
    best_solution = current_solution

    for _ in range(iterations):
        neighbor_solution = current_solution + np.random.uniform(-1, 1)
        
        current_energy = objective_function(current_solution)
        neighbor_energy = objective_function(neighbor_solution)

        if neighbor_energy < current_energy or np.random.rand() < np.exp((current_energy - neighbor_energy) / temperature):
            current_solution = neighbor_solution

        if objective_function(current_solution) < objective_function(best_solution):
            best_solution = current_solution

        temperature *= cooling_rate

    return best_solution

# Example
initial_solution = np.random.uniform(-10, 10)
temperature = 1.0
cooling_rate = 0.95
iterations = 1000

best_solution = simulated_annealing(initial_solution, objective_function, temperature, cooling_rate, iterations)

print("Best Solution:", best_solution)
print("Objective Value at Best Solution:", objective_function(best_solution))

# Plotting the objective function
x_values = np.linspace(-10, 10, 100)
y_values = objective_function(x_values)

plt.plot(x_values, y_values, label='Objective Function')
plt.scatter(best_solution, objective_function(best_solution), color='red', marker='o', label='Best Solution')
plt.xlabel('x')
plt.ylabel('Objective Value')
plt.title('Simulated Annealing')
plt.legend()
plt.show()
