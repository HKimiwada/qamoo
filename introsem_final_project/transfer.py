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
from project_utils import ProjectLogger

def run_transfer_experiment():
    logger = ProjectLogger("Option_B_Transfer")
    backend = AerSimulator(method='matrix_product_state') 
    
    repo_root = os.getcwd() 
    data_path = os.path.join(repo_root, 'data') + '/'

    target_problem = ProblemSpecification()
    target_problem.data_folder = data_path
    target_problem.num_qubits = 27
    target_problem.num_objectives = 3
    target_problem.problem_id = 1
    target_problem.num_swap_layers = 0

    # Scenarios
    src_transfer = ProblemSpecification()
    src_transfer.data_folder = data_path
    src_transfer.num_qubits = 27
    src_transfer.num_objectives = 3
    src_transfer.problem_id = 0 
    
    src_baseline = target_problem 

    scenarios = {
        "Transfer_Learning": src_transfer,
        "Baseline_Standard": src_baseline
    }
    
    final_data = {} # To store raw arrays

    for name, src_spec in scenarios.items():
        logger.log(f"=== Running Scenario: {name} ===")
        
        config = QAOAConfig()
        config.parameter_file = src_spec.problem_folder + 'JuliQAOA_angles.json'
        config.p = 1
        config.num_samples = 20
        config.shots = 100
        config.objective_weights_id = 0
        config.backend_name = "aer_simulator_mps"
        config.run_id = f"transfer_{src_spec.problem_id}_to_{target_problem.problem_id}"
        config.problem = target_problem
        
        try:
            prepare_qaoa_circuits(config, backend, overwrite_results=True)
            transpile_qaoa_circuits_parametrized(config, backend)
            batch_execute_qaoa_circuits_parametrized([config], backend)
            
            steps = range(0, config.total_num_samples + 1, 5)
            compute_hypervolume_progress(target_problem.problem_folder, config.results_folder, steps)
            
            x, y = config.progress_x_y()
            final_data[name] = y
            logger.log(f"-> {name} Max HV: {max(y) if len(y)>0 else 0}")
            
        except Exception as e:
            logger.log(f"Error in {name}: {str(e)}")

    # Save Raw Data
    logger.save_json("transfer_history.json", final_data)

    # Plot
    plt.figure(figsize=(10,6))
    for name, hv_history in final_data.items():
        steps_x = range(0, len(hv_history) * 5, 5) 
        plt.plot(steps_x, hv_history, label=name, marker='o')
    
    plt.title("Option B: Transfer Learning Utility (27 Qubits)")
    plt.xlabel("Evaluation Samples")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(logger.get_output_dir(), "option_b_plot.png")
    plt.savefig(plot_path)
    logger.log(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_transfer_experiment()