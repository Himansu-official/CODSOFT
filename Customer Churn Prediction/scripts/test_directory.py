import os

def check_directories():
    required_dirs = ['data', 'models', 'notebooks', 'results', 'scripts', 'venv']
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"Missing directory: {d}")
        else:
            print(f"Directory exists: {d}")

check_directories()
