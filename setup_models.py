import os
from pathlib import Path
import shutil

def setup_models():
    # Define paths
    base_dir = Path(__file__).parent
    models_dir = base_dir / "Models"
    notebook_dir = base_dir / "Notebook"
    
    # Create Models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Files to check/copy
    required_files = [
        ("random_forest_model_v2.joblib", "Models"),
        ("scaler.joblib", "Models")
    ]
    
    # Check and copy files
    for file_name, source_dir in required_files:
        target_path = models_dir / file_name
        source_path = notebook_dir / source_dir / file_name
        
        if not target_path.exists():
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                print(f"Copied {file_name} to Models directory")
            else:
                print(f"ERROR: Could not find {file_name} in {source_path}")
                print("Please run the training script first:")
                print("cd Notebook")
                print("python Train_rf_model.py")
                return False
    
    print("\nAll model files are in place!")
    return True

if __name__ == "__main__":
    setup_models()