# python introsem_final_project/benchmark.py
import os
import json
import networkx as nx
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
from qamoo.configs.configs import ProblemSpecification
from project_utils import ProjectLogger

class MO_MaxCut_UDP:
    def __init__(self, graph, num_objectives=3):
        self.G = graph
        self.n = len(graph.nodes)
        self.n_obj = num_objectives
        
    def fitness(self, x):
        # Continuous genes [0,1] -> Binary [0,1]
        bits = np.round(x).astype(int)
        
        cuts = np.zeros(self.n_obj)
        
        # Iterate over all edges in the combined graph
        for u, v, data in self.G.edges(data=True):
            if bits[u] != bits[v]:
                # We expect a 'weights' list now, created by our new loader
                w = data.get('weights', [0]*self.n_obj)
                
                # Sum up cuts
                for i in range(self.n_obj):
                    cuts[i] += w[i]
                    
        return -cuts # Minimize negative cut

    def get_bounds(self):
        return ([0]*self.n, [1]*self.n)
    
    def get_nobj(self):
        return self.n_obj

def load_combined_qamoo_graph(problem_spec):
    """
    Loads problem_graph_0.json, problem_graph_1.json, etc.
    and combines them into a single graph where every edge has a 
    'weights' list: [w_obj0, w_obj1, w_obj2...]
    """
    print("Loading and merging separate graph files...")
    
    # Initialize a master graph
    # We load Graph 0 first to get nodes/edges structure
    base_path = os.path.join(problem_spec.problem_folder, "problem_graph_0.json")
    with open(base_path, 'r') as f:
        data = json.load(f)
    MasterG = nx.node_link_graph(data)
    
    # Initialize 'weights' list for every edge with [weight_of_graph_0]
    for u, v, data in MasterG.edges(data=True):
        w0 = data.get('weight', 0)
        data['weights'] = [w0] # Start list
        
    # Now load the rest (1, 2, ...)
    for i in range(1, problem_spec.num_objectives):
        next_path = os.path.join(problem_spec.problem_folder, f"problem_graph_{i}.json")
        
        if os.path.exists(next_path):
            with open(next_path, 'r') as f:
                next_data = json.load(f)
            NextG = nx.node_link_graph(next_data)
            
            # Add weights from this graph to MasterG
            for u, v, data in NextG.edges(data=True):
                w_next = data.get('weight', 0)
                
                if MasterG.has_edge(u, v):
                    MasterG[u][v]['weights'].append(w_next)
                else:
                    # If edge exists in G1 but not G0, add it
                    # (This is rare for MaxCut but possible in some datasets)
                    # We must pad the previous weights with 0
                    current_len = i 
                    new_weights = [0]*current_len + [w_next]
                    MasterG.add_edge(u, v, weights=new_weights)
        else:
            print(f"Warning: {next_path} not found. Assuming 0 weights.")
            
    # Final cleanup: Ensure all edges have a weight list of correct length
    for u, v, data in MasterG.edges(data=True):
        if 'weights' not in data:
            data['weights'] = [0] * problem_spec.num_objectives
        while len(data['weights']) < problem_spec.num_objectives:
            data['weights'].append(0)
            
    return MasterG

def run_classical_benchmark():
    logger = ProjectLogger("Option_C_Benchmark_Fixed")
    
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'
    
    problem = ProblemSpecification()
    problem.data_folder = data_path
    problem.num_qubits = 27
    problem.num_objectives = 3
    problem.problem_id = 0
    
    logger.log(f"Loading {problem.num_objectives} graphs...")
    G = load_combined_qamoo_graph(problem)
    logger.log(f"Combined Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    logger.log("Running NSGA-II (Classical)...")
    udp = MO_MaxCut_UDP(G, num_objectives=problem.num_objectives)
    pop = pg.population(pg.problem(udp), size=200) # Increased pop size for better front
    algo = pg.algorithm(pg.nsga2(gen=200))         # Increased gens for convergence
    pop = algo.evolve(pop)
    
    classical_front = -np.array(pop.get_f())
    
    # Save Raw Points
    front_list = classical_front.tolist()
    logger.save_json("classical_front_points.json", {"points": front_list})
    
    # Plot (Obj 1 vs Obj 2)
    plt.figure(figsize=(8,6))
    plt.scatter(classical_front[:,0], classical_front[:,1], c='blue', label='Classical Front')
    plt.title(f"Option C: Classical Benchmark (27 Qubits)")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(logger.get_output_dir(), "option_c_plot.png")
    plt.savefig(plot_path)
    logger.log(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_classical_benchmark()