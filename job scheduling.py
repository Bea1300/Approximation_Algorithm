import numpy as np

# Function to calculate total completion time of a schedule
def total_completion_time(schedule, processing_times):
    completion_times = np.zeros(len(schedule))
    for i, machine in enumerate(schedule):
        completion_times[machine] += processing_times[i]
    return np.max(completion_times)

# Genetic Algorithm for Job Scheduling
def job_scheduling_genetic_algorithm(processing_times, num_machines, population_size=50, generations=1000, mutation_rate=0.01):
    num_jobs = len(processing_times)

    if num_machines < num_jobs:
        raise ValueError("Number of machines should be greater than or equal to the number of jobs.")

    population = [np.random.permutation(num_machines) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [1 / total_completion_time(schedule, processing_times) for schedule in population]
        selected_indices = np.random.choice(population_size, size=population_size, replace=True, p=fitness_scores/np.sum(fitness_scores))

        new_population = []

        for i in range(0, len(selected_indices) - 1, 2):
            parent1 = population[selected_indices[i]]
            parent2 = population[selected_indices[i + 1]]

            # Crossover (Order Crossover)
            crossover_point1 = np.random.randint(num_machines)
            crossover_point2 = np.random.randint(crossover_point1 + 1, num_machines + 1)

            child1 = np.concatenate((parent1[crossover_point2:], parent1[:crossover_point2]))
            child2 = np.concatenate((parent2[crossover_point2:], parent2[:crossover_point2]))

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_point1, mutation_point2 = np.random.choice(num_jobs, size=2, replace=False)
                child1[mutation_point1], child1[mutation_point2] = child1[mutation_point2], child1[mutation_point1]

            if np.random.rand() < mutation_rate:
                mutation_point1, mutation_point2 = np.random.choice(num_jobs, size=2, replace=False)
                child2[mutation_point1], child2[mutation_point2] = child2[mutation_point2], child2[mutation_point1]

            new_population.extend([child1, child2])

        population = new_population

    best_schedule = population[np.argmax([total_completion_time(schedule, processing_times) for schedule in population])]
    return best_schedule

# Example
processing_times = [4, 5, 8, 2, 6]
num_machines = 5  # Ensure this is at least equal to the number of jobs

best_schedule = job_scheduling_genetic_algorithm(processing_times, num_machines, population_size=50, generations=1000, mutation_rate=0.01)

print("Best Schedule:", best_schedule)
print("Total Completion Time:", total_completion_time(best_schedule, processing_times))
