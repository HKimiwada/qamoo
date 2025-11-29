# python introsem_final_project/transfer.py
import os
import matplotlib.pyplot as plt
from qamoo.configs.configs import ProblemSpecification, QAOAConfig
from qamoo.algorithms.qaoa import (
    prepare_qaoa_circuits, 
    transpile_qaoa_circuits_parametrized, 
    batch_execute_qaoa_circuits_parametrized
)
from qamoo.utils.utils import compute_hypervolume_progress
from qiskit_aer import AerSimulator

def run_transfer_experiment():
    # 1. Setup Backend (MPS is mandatory for 27q)
    backend = AerSimulator(method='matrix_product_state') 
    
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'

    # 2. Define The TARGET Problem (We want to solve Instance #1)
    target_problem = ProblemSpecification()
    target_problem.data_folder = data_path
    target_problem.num_qubits = 27
    target_problem.num_objectives = 3
    target_problem.problem_id = 1 # Solving ID 1
    target_problem.num_swap_layers = 0

    # 3. Define Parameter Sources
    # Source A: Transfer (Parameters from Instance #0) -> "Generalized Learning"
    src_transfer = ProblemSpecification()
    src_transfer.data_folder = data_path
    src_transfer.num_qubits = 27
    src_transfer.num_objectives = 3
    src_transfer.problem_id = 0 
    
    # Source B: Baseline (Parameters from Instance #1 itself) -> "Standard/Ideal"
    # Note: In a real "from scratch" test, we wouldn't have these, but this 
    # serves as our control group.
    src_baseline = target_problem 

    scenarios = {
        "Transfer (Params from ID 0)": src_transfer,
        "Baseline (Params from ID 1)": src_baseline
    }
    
    final_hvs = {}

    for name, src_spec in scenarios.items():
        print(f"\n=== Running Scenario: {name} ===")
        
        config = QAOAConfig()
        # CRITICAL: We load parameters from 'src_spec', but solve 'target_problem'
        config.parameter_file = src_spec.problem_folder + 'JuliQAOA_angles.json'
        
        config.p = 1
        config.num_samples = 20  # Reduced for speed
        config.shots = 100       # Reduced for speed
        config.objective_weights_id = 0
        config.backend_name = "aer_simulator_mps"
        # Unique ID to prevent folder collisions
        config.run_id = f"transfer_{src_spec.problem_id}_to_{target_problem.problem_id}"
        config.problem = target_problem # Always solving the target
        
        try:
            print(f"1. Preparing Circuits (Source: {src_spec.problem_id} -> Target: {target_problem.problem_id})...")
            prepare_qaoa_circuits(config, backend, overwrite_results=True)
            
            print("2. Transpiling...")
            transpile_qaoa_circuits_parametrized(config, backend)
            
            print("3. Executing...")
            batch_execute_qaoa_circuits_parametrized([config], backend)
            
            # Analyze Hypervolume
            steps = range(0, config.total_num_samples + 1, 5) # Frequent checks
            compute_hypervolume_progress(target_problem.problem_folder, config.results_folder, steps)
            
            x, y = config.progress_x_y()
            final_hvs[name] = y
            print(f"-> Final HV: {max(y) if len(y)>0 else 0}")
            
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    # Plot Comparison
    plt.figure(figsize=(10,6))
    for name, hv_history in final_hvs.items():
        # x-axis is just steps
        steps_x = range(0, len(hv_history) * 5, 5) 
        plt.plot(steps_x, hv_history, label=name, marker='o')
    
    plt.title("Option B: Transfer Learning Utility (27 Qubits)")
    plt.xlabel("Evaluation Samples")
    plt.ylabel("Hypervolume (Solution Quality)")
    plt.legend()
    plt.grid(True)
    plt.savefig("option_b_results.png")
    print("Saved option_b_results.png")

if __name__ == "__main__":
    run_transfer_experiment()