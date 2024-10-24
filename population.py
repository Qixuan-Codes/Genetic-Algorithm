import creature 
import genome
import numpy as np

class Population:
    # Custom Code
    def __init__(self, pop_size, gene_count):
        # Initialise the population with a list of creatures
        self.creatures = [creature.Creature(gene_count=gene_count) for i in range(pop_size)]
        self.gene_count = gene_count  # Add gene_count as an attribute

    # Custom Code
    @staticmethod
    def get_fitness_map(fits):
        # Create a fitness map for selection based on the inverted fitness value
        # The lower the fitness score the better it is
        fitmap = []
        total = 0
        max_fitness = max(fits)
        inverted_fits = [max_fitness - f for f in fits]  # Invert fitness values

        for f in inverted_fits:
            total += f
            fitmap.append(total)
        return fitmap
    
    @staticmethod
    def select_parent(fitmap):
        # Select a parent based on the fitness map
        r = np.random.rand()
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i

    # Custom Code
    def evaluate_population(self, sim, iterations, target_coords=None):
        # Evaluate the fitness of each creature in the population
        for i, cr in enumerate(self.creatures):
            if cr is None or cr.dna is None:
                # Reinitialise the creature if it's None or has no DNA
                cr = creature.Creature(gene_count=self.gene_count)
                self.creatures[i] = cr
            cr.fitness = sim.run_creature(cr, iterations, target_coords)
    
    # Custom Code
    def evolve_population(self, fit_map, mutation_rates):
        # Evolve the population by selecting parents and applying crossover and mutation
        new_creatures = []
        for _ in range(len(self.creatures)):
            p1_ind = self.select_parent(fit_map)
            p2_ind = self.select_parent(fit_map)
            p1 = self.creatures[p1_ind]
            p2 = self.creatures[p2_ind]
            dna = genome.Genome.crossover(p1.dna, p2.dna)
            dna = genome.Genome.point_mutate(dna, rate=mutation_rates[0], amount=mutation_rates[1])
            dna = genome.Genome.shrink_mutate(dna, rate=mutation_rates[2])
            dna = genome.Genome.grow_mutate(dna, rate=mutation_rates[3])
            cr = creature.Creature(gene_count=self.gene_count)
            cr.update_dna(dna)
            new_creatures.append(cr)
        self.creatures = new_creatures