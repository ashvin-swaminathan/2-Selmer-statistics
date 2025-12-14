import sys
import time
import subprocess
import itertools
from multiprocessing import Pool
from functools import lru_cache
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from networkx.utils import UnionFind

# Parameters
dimension_start = 2
dimension_end = 10
is_debug_mode = False

@lru_cache(None)
def get_vertices(dimension):
    return list(itertools.product([0, 1], repeat=dimension))

@lru_cache(None)
def get_vertex_to_index_map(dimension):
    return {vertex: index for index, vertex in enumerate(get_vertices(dimension))}

@lru_cache(None)
def get_faces_of_dimension(dimension, face_dimension):
    if face_dimension < 0 or face_dimension > dimension:
        return []
    
    vertex_to_index = get_vertex_to_index_map(dimension)
    face_list = []
    
    for varying_axes in itertools.combinations(range(dimension), face_dimension):
        fixed_axes = [axis_index for axis_index in range(dimension) if axis_index not in varying_axes]
        
        for fixed_values in itertools.product([0, 1], repeat=dimension - face_dimension):
            base_coordinate = {axis: value for axis, value in zip(fixed_axes, fixed_values)}
            face_vertices = []
            
            for varying_values in itertools.product([0, 1], repeat=face_dimension):
                coordinate = base_coordinate.copy()
                for index, axis in enumerate(varying_axes):
                    coordinate[axis] = varying_values[index]
                
                vertex_tuple = tuple(coordinate[axis_index] for axis_index in range(dimension))
                face_vertices.append(vertex_to_index[vertex_tuple])
            
            face_list.append(tuple(sorted(face_vertices)))
            
    return sorted(list(set(face_list)))

@lru_cache(None)
def get_permutations(dimension):
    vertices = get_vertices(dimension)
    vertex_to_index = get_vertex_to_index_map(dimension)
    permutation_maps = []
    
    for axis_permutation in itertools.permutations(range(dimension)):
        new_indices = [
            vertex_to_index[tuple(vertices[index][axis] for axis in axis_permutation)] 
            for index in range(len(vertices))
        ]
        permutation_maps.append(tuple(new_indices))
    return permutation_maps

def decode_graph6(graph6_string):
    dimension = ord(graph6_string[0]) - 63
    adjacency_matrix = [[0] * dimension for _ in range(dimension)]
    data_bytes = [ord(character) - 63 for character in graph6_string[1:]]
    binary_stream = []
    
    for value in data_bytes:
        for shift_amount in range(5, -1, -1):
            binary_stream.append((value >> shift_amount) & 1)
            
    stream_index = 0
    for column_index in range(1, dimension):
        for row_index in range(column_index):
            if stream_index < len(binary_stream):
                bit_value = binary_stream[stream_index]
                adjacency_matrix[row_index][column_index] = bit_value
                adjacency_matrix[column_index][row_index] = bit_value
                stream_index += 1
    return adjacency_matrix

def encode_graph6(adjacency_matrix):
    dimension = len(adjacency_matrix)
    output_characters = [chr(dimension + 63)]
    binary_stream = [adjacency_matrix[row_index][column_index] for column_index in range(1, dimension) for row_index in range(column_index)]
    
    while len(binary_stream) % 6:
        binary_stream.append(0)
        
    for index in range(0, len(binary_stream), 6):
        value = 0
        for bit in binary_stream[index : index + 6]:
            value = (value << 1) | bit
        output_characters.append(chr(value + 63))
        
    return "".join(output_characters)

def get_canonical_configuration(configuration, dimension):
    if not configuration:
        return tuple()
    all_permutations = get_permutations(dimension)
    return min(tuple(configuration[index] for index in permutation) for permutation in all_permutations)

def is_consistent_configuration(configuration, dimension):
    if None in configuration:
        return False
        
    maximum_rank = configuration.count(1)
    if any(rank > maximum_rank for rank in configuration):
        return False

    vertex_count = 1 << dimension
    disjoint_set = UnionFind(range(vertex_count))
    faces = get_faces_of_dimension(dimension, 2)

    for face in faces:
        rank_grouping = defaultdict(list)
        for vertex_index in face:
            rank_grouping[configuration[vertex_index]].append(vertex_index)
            
        if len(rank_grouping) == 2 and all(len(nodes) == 2 for nodes in rank_grouping.values()):
            minimum_rank = min(rank_grouping.keys())
            if minimum_rank + 1 in rank_grouping:
                node_a = rank_grouping[minimum_rank][0]
                node_b = rank_grouping[minimum_rank][1]
                disjoint_set.union(node_a, node_b)
                
    groups = defaultdict(list)
    for index in range(vertex_count):
        # Access by index via getitem in networkx
        root = disjoint_set[index]
        groups[root].append(index)
        
    group_ranks = {}
    for root, members in groups.items():
        base_rank = configuration[members[0]]
        group_ranks[root] = base_rank
        if any(configuration[member] != base_rank for member in members):
            return False

    parent_map = defaultdict(set)
    for face in faces:
        rank_set = {configuration[vertex_index] for vertex_index in face}
        if len(rank_set) == 3:
            minimum_rank = min(rank_set)
            if {minimum_rank, minimum_rank + 1, minimum_rank + 2} == rank_set:
                middle_nodes = [v for v in face if configuration[v] == minimum_rank + 1]
                if len(middle_nodes) != 2:
                    continue
                    
                root_one = disjoint_set[middle_nodes[0]]
                root_two = disjoint_set[middle_nodes[1]]
                
                if root_one == root_two:
                    return False 
                    
                child_nodes = [v for v in face if configuration[v] == minimum_rank + 2]
                if child_nodes:
                    root_child = disjoint_set[child_nodes[0]]
                    parent_map[root_child].add(root_one)
                    parent_map[root_child].add(root_two)
    
    for root_child, parent_roots in parent_map.items():
        rank = group_ranks[root_child]
        if rank > 0 and len(parent_roots) > (1 << rank) - 1:
            return False
            
    return True

def get_induced_subgraph(adjacency_matrix, axes_to_keep):
    sub_dimension = len(axes_to_keep)
    subgraph_matrix = [[0] * sub_dimension for _ in range(sub_dimension)]
    for new_index_i in range(sub_dimension):
        for new_index_j in range(new_index_i + 1, sub_dimension):
            original_i = axes_to_keep[new_index_i]
            original_j = axes_to_keep[new_index_j]
            value = adjacency_matrix[original_i][original_j]
            subgraph_matrix[new_index_i][new_index_j] = value
            subgraph_matrix[new_index_j][new_index_i] = value
    return subgraph_matrix

def find_isomorphism(base_configuration, dimension, target_adjacency):
    vertex_to_index = get_vertex_to_index_map(dimension)
    
    # Reconstruct the source adjacency matrix from the base configuration
    source_adjacency = [[0] * dimension for _ in range(dimension)]
    for row_index in range(dimension):
        for column_index in range(row_index + 1, dimension):
            vertex_vector = [0] * dimension
            vertex_vector[row_index] = 1
            vertex_vector[column_index] = 1
            vertex_index = vertex_to_index[tuple(vertex_vector)]
            
            if base_configuration[vertex_index] == 2:
                source_adjacency[row_index][column_index] = 1
                source_adjacency[column_index][row_index] = 1
                
    source_degrees = sorted([sum(row) for row in source_adjacency])
    target_degrees = sorted([sum(row) for row in target_adjacency])
    if source_degrees != target_degrees:
        return None
        
    mapping_array = [-1] * dimension
    used_targets = [False] * dimension
    
    def solve_recursively(current_index):
        if current_index == dimension:
            return True
        
        needed_degree = sum(source_adjacency[current_index])
        for target_index in range(dimension):
            if not used_targets[target_index] and sum(target_adjacency[target_index]) == needed_degree:
                is_valid = True
                for previous_index in range(current_index):
                    source_connected = source_adjacency[current_index][previous_index]
                    target_connected = target_adjacency[target_index][mapping_array[previous_index]]
                    if source_connected != target_connected:
                        is_valid = False
                        break
                        
                if is_valid:
                    mapping_array[current_index] = target_index
                    used_targets[target_index] = True
                    if solve_recursively(current_index + 1):
                        return True
                    used_targets[target_index] = False
        return False

    if solve_recursively(0):
        # Return Source->Target mapping directly
        return mapping_array
    return None

def construct_configuration(adjacency_matrix, dimension, configuration_cache):
    configuration = [None] * (1 << dimension)
    vertices = get_vertices(dimension)
    
    for index, vertex in enumerate(vertices):
        weight = sum(vertex)
        if weight == 0:
            configuration[index] = 0
        elif weight == 1:
            configuration[index] = 1
        elif weight == 2:
            axes = [axis for axis, value in enumerate(vertex) if value]
            has_edge = adjacency_matrix[axes[0]][axes[1]]
            configuration[index] = 2 if has_edge else 0
            
    sub_dimension = dimension - 1
    vertex_map_n = get_vertex_to_index_map(dimension)
    vertex_map_d = get_vertex_to_index_map(sub_dimension)
    sub_vertices = get_vertices(sub_dimension)
    
    for removed_axis in range(dimension):
        axes_to_keep = [axis for axis in range(dimension) if axis != removed_axis]
        subgraph_adjacency = get_induced_subgraph(adjacency_matrix, axes_to_keep)
        
        process = subprocess.Popen(["labelg", "-q", "-g"], 
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        canonical_string, _ = process.communicate(encode_graph6(subgraph_adjacency) + "\n")
        canonical_string = canonical_string.strip()
        
        if canonical_string not in configuration_cache:
            raise RuntimeError(f"Missing base {canonical_string}")
            
        base_configuration = configuration_cache[canonical_string]
        # mapping is Source -> Target
        mapping = find_isomorphism(base_configuration, sub_dimension, subgraph_adjacency)
        
        if not mapping:
            raise RuntimeError("Isomorphism failed")
            
        for sub_vertex in sub_vertices:
            old_coordinates = tuple(sub_vertex[mapping[k]] for k in range(sub_dimension))
            value = base_configuration[vertex_map_d[old_coordinates]]
            
            full_coordinates = list(sub_vertex)
            full_coordinates.insert(removed_axis, 0)
            full_index = vertex_map_n[tuple(full_coordinates)]
            
            if configuration[full_index] is None:
                configuration[full_index] = value
            elif configuration[full_index] != value:
                raise RuntimeError(f"Conflict at {full_coordinates}")

    final_index = (1 << dimension) - 1
    if configuration[final_index] is not None:
        return tuple(configuration)
        
    neighbor_indices = [final_index ^ (1 << i) for i in range(dimension)]
    
    candidates = {configuration[neighbor_indices[0]] - 1, configuration[neighbor_indices[0]] + 1}
    for neighbor_idx in neighbor_indices[1:]:
        candidates &= {configuration[neighbor_idx] - 1, configuration[neighbor_idx] + 1}
        
    valid_candidates = []
    for candidate_value in candidates:
        if candidate_value >= 0:
            configuration[final_index] = candidate_value
            if is_consistent_configuration(configuration, dimension):
                valid_candidates.append(candidate_value)
                
    if len(valid_candidates) != 1:
        raise RuntimeError(f"Ambiguous candidates: {valid_candidates}")
        
    configuration[final_index] = valid_candidates[0]
    return tuple(configuration)

def process_single_task(arguments):
    graph6_string, dimension, previous_cache = arguments
    try:
        adjacency_matrix = decode_graph6(graph6_string)
        result_configuration = construct_configuration(adjacency_matrix, dimension, previous_cache)
        return True, graph6_string, get_canonical_configuration(result_configuration, dimension)
    except Exception as error_message:
        return False, graph6_string, str(error_message)

def run_linear_programming_analysis(dimension, configurations, log_file_handle):
    if not configurations:
        return

    print(f"--Running Linear Programming for Dimension {dimension}")
    log_file_handle.write(f"Dimension {dimension}\n")
    
    matrix_equality = []
    right_hand_side_equality = []
    matrix_inequality = []
    right_hand_side_upper_bound = []
    right_hand_side_lower_bound = []
    
    for face_dim in range(1, dimension + 1):
        faces = get_faces_of_dimension(dimension, face_dim)
        face_count = len(faces)
        
        equality_constant = face_count * (1 + 2.0**(1 - face_dim))
        right_hand_side_equality.append(equality_constant)
        
        base_term = 1 + 2.0**(1 - face_dim)
        upper_bound_constant = face_count * base_term * (1 + 2.0**(2 - face_dim))
        lower_bound_constant = face_count * max(base_term**2, 6 * base_term - 8)
        right_hand_side_upper_bound.append(upper_bound_constant)
        right_hand_side_lower_bound.append(lower_bound_constant)
        
        row_equality = []
        row_inequality = []
        
        for config in configurations:
            sum_two = sum(2**min(config[v] for v in f) for f in faces)
            sum_four = sum(4**min(config[v] for v in f) for f in faces)
            row_equality.append(sum_two)
            row_inequality.append(sum_four)
            
        matrix_equality.append(row_equality)
        matrix_inequality.append(row_inequality)
        
        log_file_handle.write(f"Face Dimension={face_dim} Equality: {equality_constant}\n")
        log_file_handle.write(f"Face Dimension={face_dim} Inequality: {lower_bound_constant} <= x <= {upper_bound_constant}\n")

    number_of_constraints = dimension
    probability_divisor = 1 << dimension
    
    numpy_equality_matrix = np.array(matrix_equality)
    numpy_inequality_matrix = np.array(matrix_inequality)
    numpy_upper_bound_vector = np.array(right_hand_side_upper_bound)
    numpy_lower_bound_vector = np.array(right_hand_side_lower_bound)
    
    for target_rank in range(10):
        print(f"  Target Rank={target_rank}")
        active_variables_map = {}
        
        for config_index in range(len(configurations)):
            for shift_amount in range(target_rank + 5):
                active_variables_map[(config_index, shift_amount)] = len(active_variables_map)
                
        iteration_counter = 0
        while True:
            iteration_counter += 1
            number_of_variables = len(active_variables_map)
            objective_vector = np.zeros(number_of_variables)
            
            lp_equality_matrix = np.zeros((number_of_constraints + 1, number_of_variables))
            lp_equality_rhs = np.concatenate([right_hand_side_equality, [1]])
            
            lp_inequality_matrix = np.zeros((2 * number_of_constraints, number_of_variables))
            lp_inequality_rhs = np.concatenate([numpy_upper_bound_vector, -numpy_lower_bound_vector])
            
            for (config_idx, shift), column_idx in active_variables_map.items():
                target_value = target_rank - shift
                count_value = configurations[config_idx].count(target_value)
                objective_vector[column_idx] = count_value / probability_divisor
                
                scale_factor_two = 2**shift
                scale_factor_four = 4**shift
                
                lp_equality_matrix[:number_of_constraints, column_idx] = numpy_equality_matrix[:, config_idx] * scale_factor_two
                lp_equality_matrix[number_of_constraints, column_idx] = 1
                
                lp_inequality_matrix[:number_of_constraints, column_idx] = numpy_inequality_matrix[:, config_idx] * scale_factor_four
                lp_inequality_matrix[number_of_constraints:, column_idx] = -numpy_inequality_matrix[:, config_idx] * scale_factor_four
                
            solver_result = linprog(
                objective_vector, 
                A_ub=lp_inequality_matrix, 
                b_ub=lp_inequality_rhs, 
                A_eq=lp_equality_matrix, 
                b_eq=lp_equality_rhs,
                bounds=(0, None), 
                method='highs'
            )
            
            if not solver_result.success:
                print(f"    Failed: {solver_result.message}")
                break
                
            duals_equality = solver_result.eqlin.marginals
            duals_inequality = solver_result.ineqlin.marginals
            
            best_reduced_cost = 0
            best_new_variable = None
            next_shift_amount = max(s for _, s in active_variables_map) + 1
            
            scale_factor_two = 2**next_shift_amount
            scale_factor_four = 4**next_shift_amount
            
            for config_idx in range(len(configurations)):
                target_value = target_rank - next_shift_amount
                cost_value = configurations[config_idx].count(target_value) / probability_divisor
                
                column_equality = np.append(numpy_equality_matrix[:, config_idx] * scale_factor_two, 1)
                column_inequality = np.concatenate([
                    numpy_inequality_matrix[:, config_idx] * scale_factor_four, 
                    -numpy_inequality_matrix[:, config_idx] * scale_factor_four
                ])
                
                reduced_cost = cost_value - np.dot(duals_equality, column_equality) - np.dot(duals_inequality, column_inequality)
                if reduced_cost < best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_new_variable = (config_idx, next_shift_amount)
                    
            if best_reduced_cost >= -1e-9:
                print(f"    Done (iteration {iteration_counter}). Value: {solver_result.fun:.6f}")
                break
                
            active_variables_map[best_new_variable] = len(active_variables_map)
            if iteration_counter > 100:
                break

if __name__ == "__main__":
    start_time = time.time()
    cache_store = {1: {'@': (0, 1)}}
    
    with open("results_cubes.txt", "w") as file_output, open("results_relations.txt", "w") as file_log:
        for dimension in range(dimension_start, dimension_end + 1):
            elapsed_seconds = time.time() - start_time
            print(f"[{elapsed_seconds:.1f}s] Processing Dimension={dimension}")
            
            if (dimension - 1) not in cache_store:
                print("Missing previous dimension data")
                break
                
            shell_command = f"geng -q {dimension} | labelg -q -g"
            process_output = subprocess.check_output(shell_command, shell=True, text=True)
            unique_graphs = sorted(list(set(line.strip() for line in process_output.splitlines() if line.strip())))
            print(f"  Unique graphs: {len(unique_graphs)}")
            
            processing_tasks = [(graph, dimension, cache_store[dimension - 1]) for graph in unique_graphs]
            valid_configurations = set()
            new_cache_store = {}
            
            processed_count = 0
            with Pool() as process_pool:
                for success, graph6_string, result in process_pool.imap_unordered(process_single_task, processing_tasks):
                    processed_count += 1
                    if is_debug_mode:
                        print(f"\r{processed_count}/{len(processing_tasks)}", end="")
                        
                    if success:
                        valid_configurations.add(result)
                        new_cache_store[graph6_string] = result
                    else:
                        print(f"\nError on {graph6_string}: {result}")
            
            sorted_configurations = sorted(list(valid_configurations))
            file_output.write(f"\nDimension={dimension}\nCount: {len(sorted_configurations)}\n")
            for config in sorted_configurations:
                file_output.write(f"{config}\n")
                
            cache_store[dimension] = new_cache_store
            run_linear_programming_analysis(dimension, sorted_configurations, file_log)
