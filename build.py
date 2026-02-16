#!/usr/bin/env python
"""Auto-discovery build script for notebook-first architecture."""
import sys
from pathlib import Path
from importlib import import_module

def export_notebooks():
    """Export all notebooks in nbs/ to src/."""
    try:
        import nbdev
        nbdev.nbdev_export()
        print("✓ Exported notebooks to src/")
    except Exception as e:
        print(f"✗ nbdev export failed: {e}", file=sys.stderr)
        sys.exit(1)

def discover_and_generate_configs(src_dir="src", config_dir="configs"):
    """Recursively find all classes decorated with @lightning_config and generate YAMLs.
    
    Only generates configs for classes defined in their canonical module
    (skips re-exports to avoid duplicate YAMLs).
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

if __name__ == "__main__":
    export_notebooks()
    generated = discover_and_generate_configs()
    if generated:
        print(f"✓ Generated {len(generated)} config files:")
        for f in sorted(generated):
            print(f"  - {f}")
    else:
        print("✗ No @lightning_config classes found")
        sys.exit(1)
