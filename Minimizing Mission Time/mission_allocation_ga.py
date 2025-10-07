# ==============================================================================
#      Mission Allocation via Genetic Algorithm (Paper-Faithful Implementation)
#
# File Objective:
# This version is refactored to align more closely with the paper's description
# of the allocation problem. The GA's primary role is to evolve the *assignment*
# of GNs to UAVs. The routing within each UAV's assigned set of GNs is determined
# dynamically during fitness evaluation using a simple nearest-neighbor heuristic,
# representing a faithful implementation of the paper's high-level model.
# ==============================================================================

import numpy as np
import random
from typing import List, Dict, Tuple

import models

class MissionAllocationGA:
    """
    Manages the Genetic Algorithm to find the best mission assignment for UAVs,
    staying faithful to the paper's model.
    """

    def __init__(self, gns: np.ndarray, num_uavs: int, data_center_pos: np.ndarray, transmission_radius_d: float, params: Dict):
        """
        Initializes the GA solver.
        """
        # Problem definition
        self.gns = gns
        self.num_gns = len(gns)
        self.num_uavs = num_uavs
        self.data_center_pos = data_center_pos
        self.transmission_radius_d = transmission_radius_d

        # GA parameters from config
        self.population_size = params['GA_POPULATION_SIZE']
        self.num_iterations = params['GA_NUM_ITERATIONS']
        self.crossover_rate = params['GA_CROSSOVER_RATE']
        self.mutation_rate = params['GA_MUTATION_RATE']
        self.tournament_size = params.get('GA_TOURNAMENT_SIZE', 3) 

        # Mission cost model parameters
        self.scaling_a = params['SCALING_FACTOR_A']
        self.scaling_b = params['SCALING_FACTOR_B']

        # GA state
        self.population = []
        self.best_solution_chromosome = None
        self.best_solution_fitness = float('inf')

    def _create_initial_population(self):
        """
        Creates the initial population. Each chromosome represents an assignment of
        GNs to UAVs. Gene at index `i` holds the UAV index for GN `i`.
        """
        self.population = []
        for _ in range(self.population_size):
            # For each GN, assign a random UAV
            chromosome = [random.randint(0, self.num_uavs - 1) for _ in range(self.num_gns)]
            self.population.append(chromosome)

    def _get_routes_from_chromosome(self, chromosome: List[int]) -> Dict[int, List[int]]:
        """Decodes a chromosome into a dictionary of GN lists for each UAV."""
        routes = {i: [] for i in range(self.num_uavs)}
        for gn_idx, uav_idx in enumerate(chromosome):
            routes[uav_idx].append(gn_idx)
        return routes

    def _find_nn_tour_and_cost(self, gn_indices: List[int], start_pos: np.ndarray) -> Tuple[List[int], float]:
        """
        Calculates the tour cost for a set of GNs using a nearest-neighbor heuristic.
        This is used within the fitness function to determine the route.
        """
        if not gn_indices:
            return [], 0.0

        tour = []
        total_cost = 0.0
        
        # Create a dictionary of coordinates for the GNs in this tour
        gn_coords_map = {idx: self.gns[idx] for idx in gn_indices}
        
        # Find the first GN closest to the starting position (data center)
        current_pos = start_pos
        unvisited_indices = set(gn_indices)
        
        # Find the closest GN to the data center to start the tour
        closest_gn_idx = min(unvisited_indices, key=lambda idx: np.linalg.norm(gn_coords_map[idx] - current_pos))
        
        # Calculate cost from data center to the first GN
        total_cost += models.calculate_initial_mission_cost(
            current_pos, gn_coords_map[closest_gn_idx], self.transmission_radius_d, self.scaling_a, self.scaling_b
        )
        current_pos = gn_coords_map[closest_gn_idx]
        tour.append(closest_gn_idx)
        unvisited_indices.remove(closest_gn_idx)

        # Sequentially visit the nearest unvisited GN
        while unvisited_indices:
            next_gn_idx = min(unvisited_indices, key=lambda idx: np.linalg.norm(gn_coords_map[idx] - current_pos))
            
            total_cost += models.calculate_initial_mission_cost(
                current_pos, gn_coords_map[next_gn_idx], self.transmission_radius_d, self.scaling_a, self.scaling_b
            )
            current_pos = gn_coords_map[next_gn_idx]
            tour.append(next_gn_idx)
            unvisited_indices.remove(next_gn_idx)

        # Add the cost to return to the data center
        total_cost += models.calculate_initial_mission_cost(
            current_pos, self.data_center_pos, self.transmission_radius_d, self.scaling_a, self.scaling_b
        )
        
        return tour, total_cost

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """Evaluates fitness: the maximum mission cost among all UAVs."""
        routes = self._get_routes_from_chromosome(chromosome)
        all_tour_costs = []

        for uav_idx in range(self.num_uavs):
            gn_indices_for_uav = routes[uav_idx]
            _, tour_cost = self._find_nn_tour_and_cost(gn_indices_for_uav, self.data_center_pos)
            all_tour_costs.append(tour_cost)

        return max(all_tour_costs) if all_tour_costs else 0.0

    def _selection(self) -> List[int]:
        """Selects a parent using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        tournament_fitness = [self._calculate_fitness(ind) for ind in tournament]
        winner_index = np.argmin(tournament_fitness)
        return tournament[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Performs uniform crossover."""
        child1, child2 = parent1[:], parent2[:]
        for i in range(self.num_gns):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutation(self, chromosome: List[int]) -> List[int]:
        """Mutates a chromosome by changing the assignment of a random GN."""
        gn_to_mutate = random.randint(0, self.num_gns - 1)
        new_uav_assignment = random.randint(0, self.num_uavs - 1)
        chromosome[gn_to_mutate] = new_uav_assignment
        return chromosome

    def solve(self) -> Dict:
        """Runs the entire GA process."""
        print("Starting Genetic Algorithm for mission allocation...")
        self._create_initial_population()

        for gen in range(self.num_iterations):
            fitness_scores = [self._calculate_fitness(ind) for ind in self.population]

            min_fitness_in_gen = min(fitness_scores)
            if min_fitness_in_gen < self.best_solution_fitness:
                self.best_solution_fitness = min_fitness_in_gen
                best_idx = np.argmin(fitness_scores)
                self.best_solution_chromosome = self.population[best_idx][:]

            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{self.num_iterations}, Best Fitness: {self.best_solution_fitness:.2f}")

            next_generation = []
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

        # Decode the final best chromosome to get the assignment and the final ordered routes
        final_routes_decoded = self._get_routes_from_chromosome(self.best_solution_chromosome)
        final_assignment = {}
        for i in range(self.num_uavs):
            gn_indices = final_routes_decoded.get(i, [])
            # Get the final ordered tour for this assignment
            ordered_tour, _ = self._find_nn_tour_and_cost(gn_indices, self.data_center_pos)
            final_assignment[f"UAV_{i}"] = ordered_tour
        
        result = {
            "assignment": final_assignment,
            "min_max_cost": self.best_solution_fitness
        }
        
        print("Genetic Algorithm finished.")
        print(f"Final Min-Max Mission Cost: {result['min_max_cost']:.2f}")
        for uav, route in result['assignment'].items():
            print(f"  {uav}: Route {route}")
        return result