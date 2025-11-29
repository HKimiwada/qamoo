import os
import json
import glob

def fix_missing_bounds_robust():
    # Path to the 27q problems
    base_path = os.path.join(os.getcwd(), 'data', 'problems', '27q')
    
    if not os.path.exists(base_path):
        print(f"Error: Could not find path {base_path}")
        return

    # Find all problem folders
    problem_folders = glob.glob(os.path.join(base_path, "problem_set_*"))
    
    print(f"Found {len(problem_folders)} problem sets. Updating bounds...")

    for folder in problem_folders:
        folder_name = os.path.basename(folder)
        
        # Determine number of objectives
        if "_3o_" in folder_name:
            n_obj = 3
        elif "_4o_" in folder_name:
            n_obj = 4
        elif "_2o_" in folder_name:
            n_obj = 2
        else:
            continue

        # 1. Update lower_bounds.json
        # We use -100.0. When qamoo computes HV, it uses -(-100) = 100 as the Reference Point.
        # This covers any positive simulation artifacts up to 100.
        lb_file = os.path.join(folder, "lower_bounds.json")
        lb_data = [-100.0] * n_obj  # Robust negative value
        with open(lb_file, "w") as f:
            json.dump(lb_data, f)
        print(f"  [Updated] lower_bounds.json for {folder_name} (Set to -100)")

        # 2. Update upper_bounds.json
        # We use a large positive number to be safe
        ub_file = os.path.join(folder, "upper_bounds.json")
        ub_data = [1000.0] * n_obj
        with open(ub_file, "w") as f:
            json.dump(ub_data, f)

    print("\nDone! Bounds are now robust. Please re-run your experiment.")

if __name__ == "__main__":
    fix_missing_bounds_robust()