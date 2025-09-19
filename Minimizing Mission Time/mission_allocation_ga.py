# ==============================================================================
#                      Mission Allocation via Genetic Algorithm
#
# File Objective:
# Implements Algorithm 1 from the research paper to solve the high-level mission
# allocation problem (a variant of the Min-Max Multiple Traveling Salesman Problem).
# A Genetic Algorithm (GA) is used to determine which UAV visits which set of
# Ground Nodes (GNs) and in what initial order, aiming to balance the workload
# by minimizing the mission time of the most heavily tasked UAV.
# ==============================================================================

# Import necessary libraries
import numpy as np
import random
from typing import List, Dict, Tuple

# Import models from the local module for fitness calculation
import models

class MissionAllocationGA:
    """
    Manages the Genetic Algorithm to find the best mission assignment for UAVs.

    This class encapsulates the entire GA process, including population management,
    fitness evaluation, and evolutionary operators (selection, crossover, mutation),
    to solve the Min-Max Multiple Traveling Salesman Problem for UAV mission planning.
    """

    def __init__(self, gns: np.ndarray, num_uavs: int, data_center_pos: np.ndarray, params: Dict):
        """
        Initializes the GA solver with all necessary problem data.

        Args:
            gns (np.ndarray): The coordinates of all N Ground Nodes, shape (N, 2).
            num_uavs (int): The number of available UAVs (M).
            data_center_pos (np.ndarray): The coordinates of the data center.
            params (Dict): A configuration dictionary containing all GA and model parameters.
        """
        # Problem definition
        self.gns = gns
        self.num_gns = len(gns)
        self.num_uavs = num_uavs
        self.data_center_pos = data_center_pos

        # GA parameters
        self.population_size = params['GA_POPULATION_SIZE']
        self.num_iterations = params['GA_NUM_ITERATIONS']
        self.crossover_rate = params['GA_CROSSOVER_RATE']
        self.mutation_rate = params['GA_MUTATION_RATE']
        self.tournament_size = params.get('GA_TOURNAMENT_SIZE', 3) # A common default

        # Mission cost model parameters
        self.scaling_a = params['SCALING_FACTOR_A']
        self.scaling_b = params['SCALING_FACTOR_B']
        # This parameter would ideally be pre-calculated based on SNR thresholds.
        # For now, we'll assume it's provided or use a placeholder.
        self.transmission_radius_d = params.get('TRANSMISSION_RADIUS_D', 500.0)

        # GA state
        self.population = []
        self.best_solution_chromosome = None
        self.best_solution_fitness = float('inf')

    def _create_initial_population(self):
        """Generates the first generation of random solutions (chromosomes)."""
        base_chromosome = list(range(self.num_gns)) + list(range(-1, -self.num_uavs, -1))

        self.population = []
        for _ in range(self.population_size):
            shuffled_chromosome = random.sample(base_chromosome, len(base_chromosome))
            self.population.append(shuffled_chromosome)

    def _decode_chromosome(self, chromosome: List[int]) -> Dict[int, List[int]]:
        """Parses a chromosome into a dictionary of routes for each UAV."""
        routes = {}
        current_uav_idx = 0
        current_route = []

        for gene in chromosome:
            if gene < 0: # It's a divider
                routes[current_uav_idx] = current_route
                current_uav_idx += 1
                current_route = []
            else: # It's a GN index
                current_route.append(gene)
        
        routes[current_uav_idx] = current_route # Add the last route
        return routes

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Evaluates the fitness of a single chromosome.
        Fitness is defined as the maximum mission cost among all UAV tours.
        A lower fitness score is better.
        """
        routes = self._decode_chromosome(chromosome)
        all_tour_costs = []

        for uav_idx in range(self.num_uavs):
            tour_cost = 0.0
            route = routes.get(uav_idx, [])
            
            if not route:
                all_tour_costs.append(0.0)
                continue

            # Cost from data center to the first GN
            prev_coord = self.data_center_pos
            curr_coord = self.gns[route[0]]
            tour_cost += models.calculate_initial_mission_cost(
                prev_coord, curr_coord, self.transmission_radius_d, self.scaling_a, self.scaling_b
            )
            
            # Costs between GNs in the route
            for i in range(len(route) - 1):
                prev_coord = self.gns[route[i]]
                curr_coord = self.gns[route[i+1]]
                tour_cost += models.calculate_initial_mission_cost(
                    prev_coord, curr_coord, self.transmission_radius_d, self.scaling_a, self.scaling_b
                )

            # Cost from the last GN back to the data center
            prev_coord = self.gns[route[-1]]
            curr_coord = self.data_center_pos
            tour_cost += models.calculate_initial_mission_cost(
                prev_coord, curr_coord, self.transmission_radius_d, self.scaling_a, self.scaling_b
            )
            
            all_tour_costs.append(tour_cost)

        # The fitness is the maximum cost (the "min-max" objective)
        return max(all_tour_costs)

    def _selection(self) -> List[int]:
        """Selects a parent using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        tournament_fitness = [self._calculate_fitness(ind) for ind in tournament]
        winner_index = np.argmin(tournament_fitness)
        return tournament[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Performs Order Crossover (OX1) on two parents to create two offspring.
        """
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size

        # Select a random subsequence from parent1
        start, end = sorted(random.sample(range(size), 2))
        
        # Copy subsequence from parents to children
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill remaining spots for child1
        p2_genes = [gene for gene in parent2 if gene not in child1]
        idx = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = p2_genes[idx]
                idx += 1
        
        # Fill remaining spots for child2
        p1_genes = [gene for gene in parent1 if gene not in child2]
        idx = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = p1_genes[idx]
                idx += 1

        return child1, child2

    def _mutation(self, chromosome: List[int]) -> List[int]:
        """Performs swap mutation on a chromosome."""
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def solve(self) -> Dict:
        """
        Orchestrates the entire evolutionary process to find the best mission plan.
        """
        print("Starting Genetic Algorithm for mission allocation...")
        self._create_initial_population()

        for gen in range(self.num_iterations):
            # Calculate fitness for the entire population
            fitness_scores = [self._calculate_fitness(ind) for ind in self.population]

            # Update the best solution found so far
            min_fitness_in_gen = min(fitness_scores)
            if min_fitness_in_gen < self.best_solution_fitness:
                self.best_solution_fitness = min_fitness_in_gen
                best_idx = np.argmin(fitness_scores)
                self.best_solution_chromosome = self.population[best_idx]

            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{self.num_iterations}, Best Fitness: {self.best_solution_fitness:.2f}")

            # Create the next generation
            next_generation = []
            # Elitism: Keep the best individual from the current generation
            best_current_idx = np.argmin(fitness_scores)
            next_generation.append(self.population[best_current_idx])
            
            while len(next_generation) < self.population_size:
                parent1 = self._selection()
                parent2 = self._selection()

                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]

                if random.random() < self.mutation_rate:
                    offspring1 = self._mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutation(offspring2)
                
                next_generation.append(offspring1)
                if len(next_generation) < self.population_size:
                    next_generation.append(offspring2)

            self.population = next_generation

        # Decode the best chromosome into a human-readable format
        final_routes_decoded = self._decode_chromosome(self.best_solution_chromosome)
        
        # Format the output
        final_assignment = {
            f"UAV_{i}": final_routes_decoded.get(i, []) for i in range(self.num_uavs)
        }
        
        result = {
            "assignment": final_assignment,
            "min_max_cost": self.best_solution_fitness
        }
        
        print("Genetic Algorithm finished.")
        print(f"Final Min-Max Mission Cost: {result['min_max_cost']:.2f}")
        return result