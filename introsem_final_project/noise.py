# python introsem_final_project/noise.py
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
from qiskit_aer.noise import NoiseModel, depolarizing_error
from project_utils import ProjectLogger  # New Import

def get_noisy_backend(error_rate):
    mps_options = {"matrix_product_state_max_bond_dimension": 20} 
    if error_rate == 0:
        return AerSimulator(method='matrix_product_state', **mps_options)
        
    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_rate, 1)
    error_2q = depolarizing_error(error_rate * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'h', 'sx', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cz', 'rzz'])
    return AerSimulator(noise_model=noise_model, method='matrix_product_state', **mps_options)

def run_noise_experiment():
    # Initialize Logger
    logger = ProjectLogger("Option_A_Noise")
    
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'
    
    problem = ProblemSpecification()
    problem.data_folder = data_path
    problem.num_qubits = 27
    problem.num_objectives = 3
    problem.num_swap_layers = 0
    problem.problem_id = 0
    
    param_spec = ProblemSpecification()
    param_spec.data_folder = data_path
    param_spec.num_qubits = 27
    param_spec.num_objectives = 3
    param_spec.num_swap_layers = 0
    param_spec.problem_id = 0

    noise_levels = [0.0, 0.0001, 0.0005, 0.001] 
    results = {}

    for noise in noise_levels:
        logger.log(f"=== Running with Noise Rate: {noise} ===")
        
        backend = get_noisy_backend(noise)
        config = QAOAConfig()
        config.parameter_file = param_spec.problem_folder + 'JuliQAOA_angles.json'
        config.p = 1
        config.num_samples = 20
        config.shots = 100
        config.objective_weights_id = 0
        config.backend_name = f"sim_noise_{noise}"
        config.run_id = f"noise_exp_{noise}"
        config.problem = problem
        
        try:
            prepare_qaoa_circuits(config, backend, overwrite_results=True)
            transpile_qaoa_circuits_parametrized(config, backend)
            batch_execute_qaoa_circuits_parametrized([config], backend)
            
            steps = range(0, config.total_num_samples + 1, 10)
            compute_hypervolume_progress(problem.problem_folder, config.results_folder, steps)
            
            x, y = config.progress_x_y()
            final_hv = max(y) if len(y) > 0 else 0
            results[noise] = final_hv
            logger.log(f"-> Noise: {noise}, Final HV: {final_hv}")
            
        except Exception as e:
            logger.log(f"Error at noise {noise}: {str(e)}")

    # Save Data
    logger.save_json("noise_results.json", results)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(list(results.keys()), list(results.values()), 'ro-', linewidth=2)
    plt.title("Option A: QAOA Robustness (27 Qubits)")
    plt.xlabel("Depolarizing Error Rate")
    plt.ylabel("Hypervolume (Solution Quality)")
    plt.grid(True)
    
    plot_path = os.path.join(logger.get_output_dir(), "option_a_plot.png")
    plt.savefig(plot_path)
    logger.log(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_noise_experiment()