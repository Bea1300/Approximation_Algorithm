import numpy as np

# Function to calculate total distance of a route
def total_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    total += distance_matrix[route[-1]][route[0]]  # Return to the starting city
    return total

# Genetic Algorithm
def genetic_algorithm(distance_matrix, population_size=50, generations=1000, mutation_rate=0.01):
    num_cities = len(distance_matrix)
    population = [np.random.permutation(num_cities) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [1 / total_distance(route, distance_matrix) for route in population]
        selected_indices = np.random.choice(population_size, size=population_size, replace=True, p=fitness_scores/np.sum(fitness_scores))

        new_population = []

        for i in range(0, len(selected_indices) - 1, 2):
            parent1 = population[selected_indices[i]]
            parent2 = population[selected_indices[i + 1]]

            # Crossover (Order Crossover)
            crossover_point1 = np.random.randint(num_cities)
            crossover_point2 = np.random.randint(crossover_point1 + 1, num_cities + 1)

            child1 = np.concatenate((parent1[crossover_point2:], parent1[:crossover_point2]))
            child2 = np.concatenate((parent2[crossover_point2:], parent2[:crossover_point2]))

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_point1, mutation_point2 = np.random.choice(num_cities, size=2, replace=False)
                child1[mutation_point1], child1[mutation_point2] = child1[mutation_point2], child1[mutation_point1]

            if np.random.rand() < mutation_rate:
                mutation_point1, mutation_point2 = np.random.choice(num_cities, size=2, replace=False)
                child2[mutation_point1], child2[mutation_point2] = child2[mutation_point2], child2[mutation_point1]

            new_population.extend([child1, child2])

        population = new_population

    best_route = population[np.argmax([total_distance(route, distance_matrix) for route in population])]
    return best_route

# Example
cities = ['A', 'B', 'C', 'D']
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

best_route = genetic_algorithm(distance_matrix, population_size=50, generations=1000, mutation_rate=0.01)

print("Best Route:", [cities[i] for i in best_route])
print("Total Distance:", total_distance(best_route, distance_matrix))
