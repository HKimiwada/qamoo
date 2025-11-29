# python introsem_final_project/compare_final.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pygmo import hypervolume

# 1. Load the Classical Front
classical_json_path = "introsem_final_results/Option_C_Benchmark_Fixed_20251129_102734/classical_front_points.json"

if not os.path.exists(classical_json_path):
    print(f"Error: Could not find {classical_json_path}")
    exit()

with open(classical_json_path, 'r') as f:
    data = json.load(f)
    # These points are POSITIVE (e.g., 7.89)
    classical_points_positive = np.array(data['points'])

# 2. Calculate Classical Hypervolume (The Correct Way)
# Step A: Convert back to "Minimization" format (Negative)
# This matches what QAMOO does internally during the simulation.
classical_points_negative = -1 * classical_points_positive

# Step B: Use the same Reference Point as QAMOO
# QAMOO computes HV relative to -(-100) = +100.
ref_point = [100.0, 100.0, 100.0] 

hv_algo = hypervolume(classical_points_negative)
classical_hv = hv_algo.compute(ref_point)

print(f"Classical Hypervolume (Aligned): {classical_hv:.2f}")

# 3. Load Quantum History
quantum_folder = "Option_B_Transfer_20251129_102148"  
quantum_file = os.path.join("introsem_final_results", quantum_folder, "transfer_history.json")

if os.path.exists(quantum_file):
    with open(quantum_file, 'r') as f:
        quantum_data = json.load(f)

    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Quantum Curves
    for name, history in quantum_data.items():
        # Create x-axis steps (0, 5, 10...)
        steps = range(0, len(history) * 5, 5)
        plt.plot(steps, history, label=f"Quantum ({name})", linewidth=2)
        
    # Plot Classical Baseline (Horizontal Line)
    plt.axhline(y=classical_hv, color='black', linestyle='--', linewidth=2, label='Classical Benchmark (NSGA-II)')
    
    plt.title("Quantum vs. Classical Performance (27 Qubits)")
    plt.xlabel("Quantum Evaluations")
    plt.ylabel("Hypervolume (Higher is Better)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("final_comparison.png")
    print("Success! Saved final_comparison.png")
else:
    print(f"Could not find quantum data at: {quantum_file}")
    print("Please edit the 'quantum_folder' variable in the script to match your timestamped folder.")