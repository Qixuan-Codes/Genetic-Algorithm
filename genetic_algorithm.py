import population
import simulation
import genome
import creature
import numpy as np
import signal
import sys
import os
import json
import matplotlib.pyplot as plt
import time
import glob

def save_state(pop, filename, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature_dna):
    # Save the state of the population and relevant statistics to a JSON file
    state = {
        "iteration": int(iteration),
        "fitness_over_time": list(map(float, fitness_over_time)),
        "mean_fitness_over_time": list(map(float, mean_fitness_over_time)),
        "max_links_over_time": list(map(int, max_links_over_time)),
        "std_dev_fitness_over_time": list(map(float, std_dev_fitness_over_time)),
        "median_fitness_over_time": list(map(float, median_fitness_over_time)),
        "best_fitness": float(best_fitness),
        "best_creature_dna": best_creature_dna.tolist() if best_creature_dna is not None else None,
        "creatures": []
    }
    for cr in pop.creatures:
        state["creatures"].append({
            "dna": cr.dna.tolist()
        })
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_state(filename, gene_count):
    # Load the state of the population and relevant statistics from the JSON file
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        state = json.load(f)
    pop = population.Population(pop_size=0, gene_count=gene_count)
    for cr_state in state["creatures"]:
        dna = np.array(cr_state["dna"])
        cr = creature.Creature(gene_count)
        cr.update_dna(dna)
        pop.creatures.append(cr)

    # Set best creature from JSON file
    best_creature_dna = np.array(state["best_creature_dna"]) if state["best_creature_dna"] is not None else None
    return pop, state["iteration"], state["fitness_over_time"], state["mean_fitness_over_time"], state["max_links_over_time"], state["std_dev_fitness_over_time"], state["median_fitness_over_time"], state["best_fitness"], best_creature_dna

def save_graphs(fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, iteration):
    # Save the various graphs
    window_size = 100

    plt.figure()
    plt.plot(moving_average(fitness_over_time, window_size), label='Distance to Target (Fitness)')
    plt.xlabel(f'Iteration')
    plt.ylabel('Distance to Target (Fitness)')
    plt.title('Distance to Target (Fitness) Over Time')
    plt.legend()
    plt.savefig(f'assets/distance_to_target_fitness_over_time_{iteration}.png')
    plt.close()

    plt.figure()
    plt.plot(moving_average(mean_fitness_over_time, window_size), label='Mean Distance to Target (Fitness)')
    plt.plot(moving_average(median_fitness_over_time, window_size), label='Median Distance to Target (Fitness)', linestyle='--')
    plt.xlabel(f'Iteration')
    plt.ylabel('Distance to Target (Fitness)')
    plt.title('Mean and Median Distance to Target (Fitness) Over Time')
    plt.legend()
    plt.savefig(f'assets/mean_and_median_distance_to_target_fitness_over_time_{iteration}.png')
    plt.close()

    plt.figure()
    plt.plot(moving_average(std_dev_fitness_over_time, window_size), label='Std Dev of Distance to Target (Fitness)')
    plt.xlabel(f'Iteration')
    plt.ylabel('Std Dev')
    plt.title('Standard Deviation of Distance to Target (Fitness) Over Time')
    plt.legend()
    plt.savefig(f'assets/std_dev_distance_to_target_fitness_over_time_{iteration}.png')
    plt.close()

    plt.figure()
    plt.plot(moving_average(max_links_over_time, window_size), label='Max Links')
    plt.xlabel(f'Iteration')
    plt.ylabel('Links')
    plt.title('Max Links Over Time')
    plt.legend()
    plt.savefig(f'assets/max_links_over_time_{iteration}.png')
    plt.close()

def moving_average(data, window_size):
    # Calculate the moving average of the given dataset with a specified window size.
    if len(data) < window_size:
        return np.cumsum(data) / np.arange(1, len(data) + 1)
    return [np.mean(data[max(i - window_size + 1, 0):i + 1]) for i in range(len(data))]

def pad_dna(dna, max_length):
    # Pad the DNA sequence to a specified maximum length
    return np.pad(dna, ((0, max_length - len(dna)), (0, 0)), 'constant')

def save_creature_to_urdf(creature, filename):
    # Save the creature to URDF
    urdf_content = creature.to_xml()
    with open(filename, 'w') as f:
        f.write(urdf_content)

def main():
    if not os.path.exists('assets'):
        os.makedirs('assets')

    for file in glob.glob("assets/*.png"):
        os.remove(file)

    pop_size = 10  # Population Size
    gene_count = 4 # Number of Genes
    steps = 2400 # Number of simulation steps
    mutation_rates = (0.1, 0.05, 0.1, 0.1) # Mutation rates for the different types of mutations

    sim = simulation.Simulation()

    best_fitness = float('inf')
    best_creature = None
    target_coords = np.array([0.5, 0.0, 5.07751996])

    fitness_over_time = []
    mean_fitness_over_time = []
    max_links_over_time = []
    std_dev_fitness_over_time = []
    median_fitness_over_time = []
    start_time = time.time()
    state_file = "assets/evolution_state.json"
    fittest_csv_file = "assets/fittest_creature.csv"
    fittest_urdf_file = "assets/fittest_creature.urdf"

    def signal_handler(sig, frame):
        # Handle CTRL+C operations to save the state and graphs
        # Allows me to stop and continue the iterations
        print("\nCtrl+C captured, saving state and graphs...")
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        save_state(pop, state_file, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature.dna if best_creature is not None else None)
        save_graphs(fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, iteration)

        print(f"Elapsed Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    state = load_state(state_file, gene_count)
    if state:
        pop, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature_dna = state
        if best_creature_dna is not None:
            best_creature = creature.Creature(gene_count)
            best_creature.update_dna(best_creature_dna)
    else:
        pop = population.Population(pop_size=pop_size, gene_count=gene_count)
        iteration = 0

    while iteration < 1000:
        # Calls population.evaluate_population
        pop.evaluate_population(sim, steps, target_coords)
        # Fitness function
        fits = [cr.distance_to_target(target_coords) + (target_coords[2] - cr.get_max_height()) for cr in pop.creatures]

        links = [len(cr.get_expanded_links()) for cr in pop.creatures]

        fits = [float('inf') if np.isnan(f) else f for f in fits]

        min_fitness = np.min(fits)
        mean_fitness = np.mean(fits)
        std_dev_fitness = np.std(fits)
        median_fitness = np.median(fits)
        max_links = np.max(links)

        fitness_over_time.append(min_fitness)
        mean_fitness_over_time.append(mean_fitness)
        std_dev_fitness_over_time.append(std_dev_fitness)
        median_fitness_over_time.append(median_fitness)
        max_links_over_time.append(max_links)

        print(f"Iteration {iteration}: distance_to_target (fitness): {np.round(min_fitness, 3)}")

        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_creature = pop.creatures[np.argmin(fits)]
            genome.Genome.to_csv(best_creature.dna, fittest_csv_file)
            save_creature_to_urdf(best_creature, fittest_urdf_file)
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"New fittest creature saved to {fittest_csv_file} and {fittest_urdf_file}")
            print(f"Elapsed Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        if best_fitness <= 1.0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Target coordinates of {target_coords} reached by a creature. Stopping evolution. Elapsed Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            save_state(pop, state_file, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature.dna if best_creature is not None else None)
            save_graphs(fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, iteration)
            break

        fit_map = population.Population.get_fitness_map(fits)

        # Use the population.evolve_population function to generate new creatures
        pop.evolve_population(fit_map, mutation_rates)
        iteration += 1

        # Save the state and graphs after every 1000 iteration
        if iteration % 100 == 0:
            save_state(pop, state_file, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature.dna if best_creature is not None else None)
            save_graphs(fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, iteration)

    # Save the state and graphs after all the iterations
    save_state(pop, state_file, iteration, fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, best_fitness, best_creature.dna if best_creature is not None else None)
    save_graphs(fitness_over_time, mean_fitness_over_time, max_links_over_time, std_dev_fitness_over_time, median_fitness_over_time, iteration)


if __name__ == "__main__":
    main()