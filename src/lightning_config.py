"""Utilities for auto-generating LightningCLI-compatible configurations from class signatures."""

import inspect
from typing import get_type_hints, Any, Dict
from pathlib import Path
import yaml


def lightning_config(cls):
    """Decorator to auto-generate LightningCLI-compatible YAML from class __init__ signature.
    
    Inspects the class's __init__ method and extracts:
    - Parameter names
    - Type hints
    - Default values
    
    Generates a YAML template where:
    - Required parameters (no default) are marked with ???
    - Optional parameters show their default value
    - Type hints are included as comments
    
    Adds to the class:
    - _lightning_config: dict with parameter names and values
    - to_yaml(): method to generate YAML string
    - to_yaml_file(path): method to write YAML to disk
    
    Example:
        @lightning_config
        class MyModel(LightningModule):
            def __init__(self, net: torch.nn.Module, lr: float = 0.001):
                ...
        
        # Generate YAML
        yaml_str = MyModel.to_yaml()
        # Output:
        # net: ???  # type: Module
        # lr: 0.001  # type: float
    """
    
    sig = inspect.signature(cls.__init__)
    try:
        type_hints = get_type_hints(cls.__init__)
    except Exception:
        # Fallback if type hints can't be resolved
        type_hints = {}
    
    config_dict = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        # Get type hint if available
        param_type = type_hints.get(param_name, param.annotation)
        type_str = ""
        
        if param_type != inspect.Parameter.empty:
            # Format type name nicely
            if hasattr(param_type, '__name__'):
                type_str = f"  # type: {param_type.__name__}"
            else:
                type_str = f"  # type: {str(param_type)}"
        
        # Get default value or mark as required
        if param.default == inspect.Parameter.empty:
            # Required parameter
            value = "???"
        else:
            # Optional parameter with default
            value = param.default
        
        # Store in config dict
        if type_str:
            config_dict[param_name] = f"{value}{type_str}"
        else:
            config_dict[param_name] = value
    
    # Attach metadata to the class
    cls._lightning_config = config_dict
    cls._config_class = cls.__name__
    
    def to_yaml(as_dict: bool = False, with_class_path: bool = True) -> Any:
        """Generate YAML config template.
        
        Args:
            as_dict: If True, return dict instead of YAML string.
            with_class_path: If True, wrap in class_path + init_args (for LightningCLI).
        
        Returns:
            YAML string or dict with config template.
        """
        # Clean dict without type comments
        clean_dict = {}
        for k, v in config_dict.items():
            if isinstance(v, str) and '  # type:' in v:
                clean_dict[k] = v.split('  # type:')[0]
            else:
                clean_dict[k] = v
        
        if with_class_path:
            # Wrap in class_path + init_args for LightningCLI
            module = cls.__module__
            class_name = cls.__name__
            structured = {
                "class_path": f"{module}.{class_name}",
                "init_args": clean_dict
            }
        else:
            structured = clean_dict
        
        if as_dict:
            return structured
        
        # Build YAML string with comments
        lines = []
        if with_class_path:
            lines.append(f"class_path: {structured['class_path']}")
            lines.append("init_args:")
            for k, v in structured['init_args'].items():
                # Get original with type comment
                orig_v = config_dict.get(k, v)
                if isinstance(orig_v, str) and '  # type:' in orig_v:
                    lines.append(f"  {k}: {orig_v}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            for k, v in clean_dict.items():
                lines.append(f"{k}: {v}")
        
        return "\n".join(lines)
    
    def to_yaml_file(path: str | Path) -> Path:
        """Write YAML config to file.
        
        Args:
            path: File path to write to.
        
        Returns:
            Path object of written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_yaml())
        return path
    
    cls.to_yaml = staticmethod(to_yaml)
    cls.to_yaml_file = staticmethod(to_yaml_file)
    
    return cls
