# python introsem_final_project/benchmark.py
import os
import json
import networkx as nx
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
from qamoo.configs.configs import ProblemSpecification

# 1. Define Classical Problem Wrapper
class MO_MaxCut_UDP:
    def __init__(self, graph, num_objectives=3):
        self.G = graph
        self.n = len(graph.nodes)
        self.n_obj = num_objectives
        
    def fitness(self, x):
        # Convert continuous [0,1] genes to binary [0,1]
        bits = np.round(x).astype(int)
        
        # Calculate cuts for each objective
        cuts = np.zeros(self.n_obj)
        
        for u, v, data in self.G.edges(data=True):
            if bits[u] != bits[v]:
                # In QAMOO 27q dataset, weights are often stored in 'weights' list: [w0, w1, w2]
                # We handle both list format and key format just in case
                if 'weights' in data:
                    w = data['weights'] 
                else:
                    # Fallback to explicit keys if list is missing
                    w = [
                        data.get('weight', 0), 
                        data.get('weight_1', 0), 
                        data.get('weight_2', 0),
                        data.get('weight_3', 0)
                    ]
                
                # Sum up cuts for valid objectives
                for i in range(min(len(w), self.n_obj)):
                    cuts[i] += w[i]
        
        return -cuts # Return negative because Pygmo minimizes

    def get_bounds(self):
        # Problem bounds: 0.0 to 1.0 for each variable
        return ([0]*self.n, [1]*self.n)
    
    def get_nobj(self):
        return self.n_obj

def load_qamoo_graph(problem_spec):
    # Load the graph structure
    graph_path = problem_spec.problem_folder + "problem_graph_0.json"
    
    print(f"Loading graph topology from: {graph_path}")
    try:
        with open(graph_path, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        return G
    except Exception as e:
        print(f"Error loading graph: {e}")
        print("Falling back to random graph (for debugging only)")
        return nx.erdos_renyi_graph(problem_spec.num_qubits, 0.5)

def run_classical_benchmark():
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'
    
    # Setup spec to locate the correct folder
    problem = ProblemSpecification()
    problem.data_folder = data_path
    problem.num_qubits = 27
    problem.num_objectives = 3
    problem.problem_id = 0
    
    # 1. Load Graph
    G = load_qamoo_graph(problem)
    print(f"Graph Loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    
    # 2. Run NSGA-II
    print("Running NSGA-II (Classical Genetic Algorithm)...")
    udp = MO_MaxCut_UDP(G, num_objectives=problem.num_objectives)
    
    # Create Population
    pop = pg.population(pg.problem(udp), size=100)
    
    # Evolve (100 generations is usually enough for 27q)
    algo = pg.algorithm(pg.nsga2(gen=100)) 
    pop = algo.evolve(pop)
    
    # Extract Pareto Front
    fits = pop.get_f()
    classical_front = -fits # Invert back to positive scores
    
    # 3. Plotting
    # Since we have 3 objectives, we plot Obj 1 vs Obj 2
    plt.figure(figsize=(8,6))
    plt.scatter(classical_front[:,0], classical_front[:,1], c='blue', label='Classical Front (Obj 1 vs 2)')
    
    plt.title(f"Option C: Classical Benchmark (27 Qubits, 3 Objectives)")
    plt.xlabel("Objective 1 (Cut Weight)")
    plt.ylabel("Objective 2 (Cut Weight)")
    plt.legend()
    plt.grid(True)
    plt.savefig("option_c_results.png")
    print("Saved option_c_results.png (2D projection of 3D front)")

if __name__ == "__main__":
    run_classical_benchmark()