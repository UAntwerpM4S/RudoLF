import numpy as np

from shapely.geometry import Point, Polygon


def attractive_force(ship_pos, target_pos, attractive_strength=5.0):
    """Calculate the attractive force toward the target position."""
    direction = np.array(target_pos) - np.array(ship_pos)
    distance = np.linalg.norm(direction)
    force = attractive_strength * direction / distance if distance != 0 else np.zeros_like(direction)
    return force

def repulsive_force(ship_pos, obstacle, repulsive_strength=10.0, obstacle_radius=100.0):
    """Calculate the repulsive force from a single polygon obstacle."""
    poly = Polygon(obstacle)
    closest_point = poly.exterior.interpolate(poly.exterior.project(Point(ship_pos)))
    closest_point_coords = np.array([closest_point.x, closest_point.y])
    distance_to_obstacle = np.linalg.norm(np.array(ship_pos) - closest_point_coords)

    if distance_to_obstacle < obstacle_radius:
        direction = np.array(ship_pos) - closest_point_coords
        repulsion_magnitude = repulsive_strength / (distance_to_obstacle + 1e-6)  # Avoid division by zero
        repulsing_force = repulsion_magnitude * direction / np.linalg.norm(direction)
        return repulsing_force, closest_point_coords
    return np.zeros(2), None

def tangential_force(ship_pos, closest_point):
    """Calculate the tangential force to slide along the obstacle boundary."""
    direction = np.array(ship_pos) - closest_point
    tangent = np.array([-direction[1], direction[0]])  # Perpendicular vector
    return tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) != 0 else np.zeros_like(tangent)

def apf(ship_pos, target_pos, obstacle, step_size=1.0, max_iterations=1000):
    """Artificial Potential Field algorithm to find the path from ship_pos to target_pos."""
    path = [ship_pos]
    poly = Polygon(obstacle)
    print(poly[0])
    
    for _ in range(max_iterations):
        # Calculate attractive force
        attractive = attractive_force(ship_pos, target_pos)
        
        # Calculate repulsive and tangential forces
        repulsive, closest_point = repulsive_force(ship_pos, obstacle)
        tangential = tangential_force(ship_pos, closest_point) if closest_point is not None else np.zeros(2)
        
        # Total force is the sum of attractive, repulsive, and tangential forces
        total_force = attractive + repulsive + 0.5 * tangential
        
        # Move the ship in the direction of the total force
        new_ship_pos = ship_pos + step_size * total_force
        
        # Ensure the ship stays within the polygon
        if not poly.contains(Point(new_ship_pos)):
            closest_point = poly.exterior.interpolate(poly.exterior.project(Point(ship_pos)))
            new_ship_pos = np.array([closest_point.x, closest_point.y])
        
        # Append the new position to the path
        path.append(new_ship_pos)
        ship_pos = new_ship_pos
        
        # Check if the ship has reached the target
        if np.linalg.norm(np.array(ship_pos) - np.array(target_pos)) < step_size:
            break
    
    return np.array(path)

def initialize_population(start, target, population_size, safe_polygon):
    """Randomly initialize a population of paths."""
    population = []
    for _ in range(population_size):
        num_waypoints = np.random.randint(3, 10)  # Random number of waypoints
        waypoints = []
        for _ in range(num_waypoints):
            while True:
                point = np.random.uniform(0, 20000, size=2)  # Random point in 2D space
                if safe_polygon.contains(Point(point)):  # Ensure point is inside the safe polygon
                    waypoints.append(point)
                    break
        path = [start] + waypoints + [target]
        population.append(path)
    return population

def fitness_function(path, safe_polygon):
    """Evaluate the fitness of a path."""
    path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
    penalty = 0
    for point in path:
        if not safe_polygon.contains(Point(point)):
            penalty += 1000  # High penalty for leaving the safe area
    smoothness = sum(np.linalg.norm(np.array(path[i + 1]) - 2 * np.array(path[i]) + np.array(path[i - 1])) 
                     for i in range(1, len(path) - 1))
    return path_length + penalty + smoothness

def select_parents(population, fitness_scores):
    """Select parents based on their fitness scores using roulette wheel selection."""
    total_fitness = sum(fitness_scores)
    probabilities = [1 - score / total_fitness for score in fitness_scores]
    probabilities = probabilities / np.sum(probabilities)  # Normalize to make a probability distribution
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    split_point = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:split_point] + parent2[split_point:]
    child2 = parent2[:split_point] + parent1[split_point:]
    return child1, child2

def mutate(path, safe_polygon, mutation_rate=0.1):
    """Mutate a path by randomly shifting waypoints or adding/removing waypoints."""
    new_path = []
    for point in path:
        if np.random.rand() < mutation_rate:
            while True:
                mutation = np.random.uniform(-500, 500, size=2)  # Small random shift
                new_point = np.array(point) + mutation
                if safe_polygon.contains(Point(new_point)):
                    new_path.append(new_point)
                    break
        else:
            new_path.append(point)
    return new_path

def genetic_algorithm(start, target, safe_polygon, population_size=50, generations=1000):
    """Find the optimal path using a genetic algorithm."""
    # Initialize population
    safe_polygon = Polygon(safe_polygon)
    population = initialize_population(start, target, population_size, safe_polygon)
    
    for generation in range(generations):
        if generation % 1000 == 0:
            print("{}/{}".format(generation, generations))
        # Evaluate fitness
        fitness_scores = [fitness_function(path, safe_polygon) for path in population]
        
        # Print best fitness in the current generation
        #print(f"Generation {generation + 1}: Best Fitness = {min(fitness_scores):.2f}")
        
        # Select parents
        selected_parents = select_parents(population, fitness_scores)
        
        # Generate offspring through crossover and mutation
        offspring = []
        for i in range(0, len(selected_parents), 2):
            if i + 1 < len(selected_parents):
                child1, child2 = crossover(selected_parents[i], selected_parents[i + 1])
                offspring.append(mutate(child1, safe_polygon))
                offspring.append(mutate(child2, safe_polygon))
        
        # Replace the old population with the new one
        population = offspring
    
    # Return the best path
    fitness_scores = [fitness_function(path, safe_polygon) for path in population]
    best_index = np.argmin(fitness_scores)
    return population[best_index]
