import os

folders = [
    "data",
    "scripts",
    "tests",
    "configs",
    "results"
]

files_to_create = {
    "scripts": [
        "data_loader.py",
        "preprocessing.py",
        "label_generation.py",
        "feature_engineering.py",
        "model_training.py",
        "evaluation.py",
        "llm_prediction.py",
        "utils.py"
    ],
    "tests": [
        "test_preprocessing.py",
        "test_label_generation.py",
        "test_model_training.py"
    ],
    "configs": [
        "experiment_config.yaml"
    ],
    "data": [
        "README.md"
    ],
    "results": [
        "README.md"
    ]
}

root_files = {
    "main.py": "# Master pipeline script for experiment\n\nif __name__ == '__main__':\n    pass\n",
    "requirements.txt": "# Add required packages here\n"
}

# Create all folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created folder: {folder}")

# Create files inside folders
for folder, files in files_to_create.items():
    for filename in files:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            with open(path, "w") as f:
                if filename.endswith(".md"):
                    f.write(f"# {folder.upper()} DIRECTORY\n\nAuto-generated. Explain how to use this folder.\n")
                else:
                    f.write(f"# {filename.replace('.py', '').replace('_', ' ').title()}\n")
            print(f"📄 Created: {path}")

# Create main.py and requirements.txt at the root
for filename, content in root_files.items():
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"🚀 Created: {filename}")