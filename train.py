import sys
import importlib
import os

EXPERIMENTS_DIR =  "experiments"

def get_available_experiments():
    """List all experiment directories inside 'experiments' folder"""
    available = []
    if not (os.path.exists(EXPERIMENTS_DIR) and os.path.isdir(EXPERIMENTS_DIR)):
        raise ValueError(f"'{EXPERIMENTS_DIR}' folder doesn't exist")
    
    for entry in os.listdir(EXPERIMENTS_DIR):
        full_path = os.path.join(EXPERIMENTS_DIR, entry)
        if os.path.isdir(full_path) and not entry.startswith("."):
            # Check if it contains train.py to be valid
            if os.path.exists(os.path.join(full_path, "train.py")):
                available.append(entry)
                
    return available

def main():
    available_experiments = get_available_experiments()
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <experiment_name> [experiment args]")
        print(f"Available experiments:")
        for experiment in available_experiments:
            print(f"\t- {experiment}")
        return
    
    experiment = sys.argv[1]
    sys.argv = [f"./train.py"] + sys.argv[2:]
    try:
        experiment_module = importlib.import_module(f"{EXPERIMENTS_DIR}.{experiment}.train")
        experiment_module.main()
    except ImportError:
        print(f"Error: Experiment '{experiment}' not found")
        print(f"Available experiments:")
        for experiment in available_experiments:
            print(f"\t- {experiment}")

if __name__ == "__main__":
    main()
    