#!/usr/bin/env python
"""Auto-discovery build script for notebook-first architecture."""
import sys
import shutil
from pathlib import Path
from importlib import import_module
import yaml

def export_notebooks():
    """Export all notebooks in nbs/ to src/."""
    try:
        from nbdev.export import nb_export
        
        nbs_dir = Path("nbs")
        for nb_file in sorted(nbs_dir.glob("*.ipynb")):
            try:
                nb_export(str(nb_file))
            except Exception as e:
                print(f"✗ Failed to export {nb_file.name}: {e}", file=sys.stderr)
                raise
        print("✓ Exported notebooks to src/")
    except Exception as e:
        print(f"✗ nbdev export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def discover_and_generate_configs(src_dir="src", config_dir="configs"):
    """Recursively find all classes decorated with @lightning_config and generate YAMLs.
    
    Only generates configs for classes defined in their canonical module
    (skips re-exports to avoid duplicate YAMLs).
    
    Note: @training_config functions are skipped - train.yaml is hand-written.
    """
    src_path = Path(src_dir)
    config_path = Path(config_dir)
    config_path.mkdir(exist_ok=True)
    
    generated = []
    seen = {}  # Track {class_name: module_where_defined}
    
    # First pass: identify where each class is actually defined
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
            
        relative = py_file.relative_to(src_path.parent)
        module_name = str(relative.with_suffix("")).replace("/", ".").replace("\\", ".")
        
        try:
            module = import_module(module_name)
            for name in dir(module):
                obj = getattr(module, name)
                # Skip functions (training_config) - only process classes (lightning_config)
                if hasattr(obj, "_lightning_config") and isinstance(obj, type):
                    # Check if this class is defined in this module (not imported)
                    if hasattr(obj, "__module__") and obj.__module__ == module_name:
                        seen[name] = (module_name, py_file.stem, obj)
        except Exception as e:
            pass
    
    # Second pass: generate configs only for canonical definitions
    for class_name, (module_name, notebook_name, cls) in seen.items():
        config_file = config_path / f"{notebook_name}.yaml"
        cls.to_yaml_file(str(config_file))
        generated.append(f"{config_file}")
    
    return generated

def generate_training_config(config_dir="configs"):
    """Discover @training_config decorated functions and generate train.yaml.
    
    Generates train.yaml with file references to component configs.
    LightningCLI will compose them at runtime.
    """
    try:
        from src.train import main
        
        if hasattr(main, '_training_config'):
            composition = main._training_config.copy()
            config_path = Path(config_dir)
            train_yaml_path = config_path / "train.yaml"
            
            # Write composition with file references as-is (no deep merging)
            with open(train_yaml_path, 'w') as f:
                yaml.dump(composition, f, default_flow_style=False, sort_keys=False)
            
            return str(train_yaml_path)
    except Exception as e:
        print(f"✗ Failed to generate train.yaml: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    return None

def generate_classpaths_config(src_dir="src", config_dir="configs"):
    """Generate train_classpaths.yaml mapping config files to their class paths.
    
    Maps from config file names (e.g., "mnist_litmodule") to full class paths
    for use by LightningCLI.
    """
    src_path = Path(src_dir)
    config_path = Path(config_dir)
    
    classpaths = {}
    
    # Find all classes decorated with @lightning_config (not @training_config functions)
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
            
        relative = py_file.relative_to(src_path.parent)
        module_name = str(relative.with_suffix("")).replace("/", ".").replace("\\", ".")
        
        try:
            module = import_module(module_name)
            for name in dir(module):
                obj = getattr(module, name)
                # Only process classes (lightning_config), skip functions (training_config)
                if hasattr(obj, "_lightning_config") and isinstance(obj, type):
                    # Check if this class is defined in this module (not imported)
                    if hasattr(obj, "__module__") and obj.__module__ == module_name:
                        # Use the notebook stem as config file key
                        config_key = py_file.stem
                        full_class_path = f"{module_name}.{name}"
                        classpaths[config_key] = full_class_path
        except Exception as e:
            pass
    
    # Write classpaths mapping
    classpaths_yaml_path = config_path / "train_classpaths.yaml"
    with open(classpaths_yaml_path, 'w') as f:
        yaml.dump(classpaths, f, default_flow_style=False, sort_keys=False)
    
    return str(classpaths_yaml_path)

if __name__ == "__main__":
    export_notebooks()
    generated = discover_and_generate_configs()
    
    # Generate train.yaml from @training_config decorator
    train_config = generate_training_config()
    if train_config:
        generated.append(train_config)
    
    # Generate train_classpaths.yaml mapping for LightningCLI
    classpaths_config = generate_classpaths_config()
    if classpaths_config:
        generated.append(classpaths_config)
    
    if generated:
        print(f"✓ Generated {len(generated)} config files:")
        for f in sorted(generated):
            print(f"  - {f}")
    else:
        print("✗ No configs generated")
        sys.exit(1)
