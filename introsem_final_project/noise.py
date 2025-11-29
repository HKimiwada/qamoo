# python introsem_final_project/noise.py
import os
import matplotlib.pyplot as plt
import numpy as np
from qamoo.configs.configs import ProblemSpecification, QAOAConfig
from qamoo.algorithms.qaoa import (
    prepare_qaoa_circuits, 
    transpile_qaoa_circuits_parametrized, 
    batch_execute_qaoa_circuits_parametrized
)
from qamoo.utils.utils import compute_hypervolume_progress
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def get_noisy_backend(error_rate):
    """Creates an MPS backend with specific depolarization error."""
    # BASE SIMULATOR: We must use matrix_product_state for 27 qubits
    if error_rate == 0:
        return AerSimulator(method='matrix_product_state')
        
    noise_model = NoiseModel()
    # Add error to 1-qubit and 2-qubit gates
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate * 10, 2)
    
    # Add noise only to standard gates to avoid MPS issues
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'h', 'sx', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cz', 'rzz'])
    
    return AerSimulator(noise_model=noise_model, method='matrix_product_state')

def run_noise_experiment():
    # 1. Setup Problem (Using 27q Data that exists)
    # NOTE: We use absolute path to avoid ".." confusion
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'
    
    problem = ProblemSpecification()
    problem.data_folder = data_path
    problem.num_qubits = 27     # UPDATED: Matches your data folder
    problem.num_objectives = 3  # UPDATED: Matches '3o' in folder name
    problem.num_swap_layers = 0
    problem.problem_id = 0
    
    # 2. Setup Parameter Source
    param_spec = ProblemSpecification()
    param_spec.data_folder = data_path
    param_spec.num_qubits = 27
    param_spec.num_objectives = 3
    param_spec.num_swap_layers = 0
    param_spec.problem_id = 0

    noise_levels = [0.0, 0.0001, 0.0005] # Kept very small for MPS stability
    results = {}

    for noise in noise_levels:
        print(f"\n=== Running with Noise Rate: {noise} ===")
        
        # A. Define Backend
        backend = get_noisy_backend(noise)
        
        # B. Configure Algorithm
        config = QAOAConfig()
        config.parameter_file = param_spec.problem_folder + 'JuliQAOA_angles.json'
        config.p = 1
        config.num_samples = 20  # REDUCED: 20 samples to save time
        config.shots = 100       # REDUCED: 100 shots to save time
        config.objective_weights_id = 0
        config.backend_name = f"sim_noise_{noise}"
        config.run_id = f"noise_exp_{noise}"
        config.problem = problem
        
        # C. Run Pipeline
        print("1. Preparing Circuits...")
        prepare_qaoa_circuits(config, backend, overwrite_results=True)
        
        print("2. Transpiling...")
        transpile_qaoa_circuits_parametrized(config, backend)
        
        print("3. Executing...")
        # Note: This might take 1-5 mins per loop due to MPS + Noise
        batch_execute_qaoa_circuits_parametrized([config], backend)
        
        # D. Analyze
        steps = range(0, config.total_num_samples + 1, 10) # Smaller steps
        compute_hypervolume_progress(problem.problem_folder, config.results_folder, steps)
        
        # Extract Final HV
        x, y = config.progress_x_y()
        final_hv = max(y) if len(y) > 0 else 0
        results[noise] = final_hv
        print(f"-> Final HV: {final_hv}")

    # Plot
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), 'ro-')
    plt.title("Option A: QAOA Robustness (27 Qubits)")
    plt.xlabel("Error Rate")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.savefig("option_a_results.png")
    print("Saved option_a_results.png")

if __name__ == "__main__":
    run_noise_experiment()