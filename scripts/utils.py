# Utils
import yaml
import os

def load_config(path="configs/experiment_config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Function to load hyperparameters from a YAML file
def load_hyperparameters(path: str = "configs/hyperparameters.yaml") -> dict:
    """
    Loads hyperparameters from a YAML file.
    Returns an empty dict if the file is not found (with a warning).
    Raises an error if the file is malformed.
    """
    if not os.path.exists(path):
        print(f"⚠️ Warning: Hyperparameter file not found at {path}. Returning empty dict.")
        return {}
    try:
        with open(path, "r") as f:
            hyperparameters = yaml.safe_load(f)
            if hyperparameters is None: # Handles empty file case
                print(f"⚠️ Warning: Hyperparameter file at {path} is empty. Returning empty dict.")
                return {}
            print(f"✅ Hyperparameters loaded successfully from {path}")
            return hyperparameters
    except yaml.YAMLError as e:
        print(f"❌ Error: Malformed YAML in hyperparameter file at {path}: {e}")
        raise # Re-raise the error as this is critical
    except Exception as e:
        print(f"❌ Error: Could not load hyperparameter file at {path}: {e}")
        raise # Re-raise other critical errors