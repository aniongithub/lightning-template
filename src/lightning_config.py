"""Utilities for auto-generating LightningCLI-compatible configurations from class signatures."""

import inspect
import importlib
from typing import get_type_hints, Any, Dict
from pathlib import Path
import yaml


def lightning_config(cls=None, *, init_args: Dict[str, Any] = None):
    """Decorator to auto-generate LightningCLI-compatible YAML from class __init__ signature.
    
    Inspects the class's __init__ method and extracts:
    - Parameter names
    - Type hints
    - Default values
    
    Generates a YAML template where:
    - Required parameters (no default) are marked with ???
    - Optional parameters show their default value
    - Type hints are included as comments
    
    Args:
        cls: The class to decorate (when used without arguments)
        init_args: Dict of parameter names to their full config structures
                  (e.g., nested class_path + init_args for component classes)
    
    Adds to the class:
    - _lightning_config: dict with parameter names and values
    - to_yaml(): method to generate YAML string
    - to_yaml_file(path): method to write YAML to disk
    
    Example:
        @lightning_config
        class SimpleModel(LightningModule):
            def __init__(self, net: torch.nn.Modu   le, lr: float = 0.001):
                ...
        
        @lightning_config(init_args={
            "net": {
                "class_path": "src.models.simpledense.SimpleDenseNet",
                "init_args": {"input_size": 784, "output_size": 10}
            }
        })
        class ConfiguredModel(LightningModule):
            def __init__(self, net: torch.nn.Module, lr: float = 0.001):
                ...
    """
    init_args = init_args or {}
    
    def decorator(cls_to_decorate):
        sig = inspect.signature(cls_to_decorate.__init__)
        try:
            type_hints = get_type_hints(cls_to_decorate.__init__)
        except Exception:
            # Fallback if type hints can't be resolved
            type_hints = {}
        
        config_dict = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Check if we have init_args for this parameter
            if param_name in init_args:
                value = init_args[param_name]
            elif param.default == inspect.Parameter.empty:
                # Required parameter with no init_args override
                value = "???"
            else:
                # Optional parameter with default
                value = param.default
            
            config_dict[param_name] = value
        
        # Define methods for the class
        def to_yaml(as_dict: bool = False, with_class_path: bool = True) -> Any:
            """Generate YAML config template."""
            # Clean dict - handle nested dicts
            clean_dict = {}
            for k, v in config_dict.items():
                if isinstance(v, str) and '  # type:' in v:
                    clean_dict[k] = v.split('  # type:')[0]
                else:
                    clean_dict[k] = v
            
            # Convert tuples to lists for OmegaConf compatibility
            def tuples_to_lists(value):
                if isinstance(value, tuple):
                    return [tuples_to_lists(v) for v in value]
                elif isinstance(value, dict):
                    return {k: tuples_to_lists(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [tuples_to_lists(v) for v in value]
                return value
            
            clean_dict = tuples_to_lists(clean_dict)
            
            if with_class_path:
                # Wrap in class_path + init_args for LightningCLI
                module = cls_to_decorate.__module__
                class_name = cls_to_decorate.__name__
                structured = {
                    "class_path": f"{module}.{class_name}",
                    "init_args": clean_dict
                }
            else:
                structured = clean_dict
            
            if as_dict:
                return structured
            
            # Return YAML string
            import yaml
            return yaml.dump(structured, default_flow_style=False, sort_keys=False)
        
        def to_yaml_file(path: str | Path) -> Path:
            """Write YAML config to file (without outer class_path wrapper for component configs)."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(to_yaml(with_class_path=False))
            return path
        
        # Attach metadata and methods to the class
        cls_to_decorate._lightning_config = config_dict
        cls_to_decorate._config_class = cls_to_decorate.__name__
        cls_to_decorate.to_yaml = staticmethod(to_yaml)
        cls_to_decorate.to_yaml_file = staticmethod(to_yaml_file)
        
        return cls_to_decorate
    
    # Support both @lightning_config and @lightning_config(init_args={...})
    if cls is None:
        # Called with arguments: @lightning_config(init_args={...})
        return decorator
    else:
        # Called without arguments: @lightning_config
        return decorator(cls)


def training_config(**composition):
    """Decorator for training orchestration that generates train.yaml and wraps main().
    
    Args:
        **composition: Keys are config file paths (seed_everything, trainer, model, data)
                      trainer and seed_everything have defaults; model and data must be specified.
        
    Example:
        @training_config(
            model="mnist_litmodule.yaml",
            data="mnist_datamodule.yaml"
        )
        def main(args=None):
            from src.train_utils import run_training
            return run_training(args)
    """
    # Set defaults only for trainer and seed_everything
    defaults = {
        'seed_everything': 42,
        'trainer': 'trainer/default.yaml',
    }
    composition = {**defaults, **composition}
    
    def decorator(func):
        # Store composition for build.py to find and generate train.yaml
        func._training_config = composition
        
        # Wrap the function to handle LightningCLI instantiation
        def wrapper(args=None):
            # Call the original function with args parameter
            return func(args=args)
        
        wrapper._training_config = composition
        return wrapper
    
    return decorator

def model(component_class_path: str):
    """Decorator for classes with required parameters satisfied by another component.
    
    Inspects the component class to extract its defaults and automatically constructs
    the full init_args structure. This enables cleaner code like:
        @model("src.models.simpledense.SimpleDenseNet")
        class MyLitModule(LightningModule):
            def __init__(self, net: nn.Module, lr: float = 0.001):
                ...
    
    Args:
        component_class_path: Full module path to the component class (e.g., 
                            "src.models.simpledense.SimpleDenseNet")
    
    Returns:
        A decorator that applies lightning_config with auto-generated init_args
    """
    def decorator(cls):
        # Dynamically import the component class
        module_path, class_name = component_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        component_cls = getattr(module, class_name)
        
        # Inspect component's __init__ to extract parameter defaults
        sig = inspect.signature(component_cls.__init__)
        component_init_args = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get default value if it exists
            if param.default != inspect.Parameter.empty:
                component_init_args[param_name] = param.default
        
        # Construct the init_args dict with component info
        init_args = {
            "net": {
                "class_path": component_class_path,
                "init_args": component_init_args
            }
        }
        
        # Apply lightning_config with the constructed init_args
        return lightning_config(cls, init_args=init_args)
    
    return decorator