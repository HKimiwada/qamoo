import os
import json
import logging
from datetime import datetime
import numpy as np

class ProjectLogger:
    def __init__(self, experiment_name):
        # Create a timestamped directory for results
        repo_root = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(repo_root, "final_results", f"{experiment_name}_{timestamp}")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Setup Logging to file and console
        log_file = os.path.join(self.output_dir, "experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Initialized experiment: {experiment_name}")
        self.logger.info(f"Saving results to: {self.output_dir}")

    def log(self, message):
        self.logger.info(message)

    def save_json(self, filename, data):
        """Saves dictionary or list data to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Helper to convert numpy types to python types for JSON
        def default(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=default)
        
        self.logger.info(f"Data saved to {filename}")
        return filepath

    def get_output_dir(self):
        return self.output_dir