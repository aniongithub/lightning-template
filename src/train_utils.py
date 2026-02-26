"""Training utilities for both CLI and notebook execution modes."""

import importlib
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer


def is_notebook():
    """Detect if running in Jupyter notebook or IPython."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False      # Standard Python interpreter


def _import_class(class_path):
    """Dynamically import a class from a class_path string.
    
    Args:
        class_path: Full class path (e.g., 'src.modules.mnist_litmodule.MNISTLitModule')
    
    Returns:
        The imported class
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_class_path(config):
    """Recursively instantiate classes from class_path + init_args."""
    if isinstance(config, dict):
        if 'class_path' in config:
            # This is a class to instantiate
            cls = _import_class(config['class_path'])
            init_args = config.get('init_args', {})
            # Recursively instantiate nested components
            init_args = {k: instantiate_class_path(v) for k, v in init_args.items()}
            return cls(**init_args)
        else:
            # Regular dict, recursively process values
            return {k: instantiate_class_path(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [instantiate_class_path(item) for item in config]
    else:
        return config


def load_and_prepare_config(args):
    """Load YAML configs and merge with overrides.
    
    Args:
        args: Optional dict of overrides. If None, parses from sys.argv
    
    Returns:
        OmegaConf config with all referenced files loaded
    """
    import sys
    
    # Find workspace root
    config_path = Path("configs/train.yaml")
    if not config_path.exists():
        config_path = Path("..") / "configs/train.yaml"
    
    config = OmegaConf.load(config_path)
    
    # Load referenced config files
    config_dir = config_path.parent
    if isinstance(config.model, str):
        config.model = OmegaConf.load(config_dir / config.model)
    if isinstance(config.data, str):
        config.data = OmegaConf.load(config_dir / config.data)
    if isinstance(config.trainer, str):
        config.trainer = OmegaConf.load(config_dir / config.trainer)
    
    # Parse CLI overrides from sys.argv if args is None
    if args is None and not is_notebook():
        cli_overrides = {}
        i = 1  # Skip program name
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith('--'):
                key = arg[2:]
                if '=' in key:
                    k, v = key.split('=', 1)
                    _nested_set(cli_overrides, k, v)
                else:
                    # Try next argument as value
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                        v = sys.argv[i + 1]
                        _nested_set(cli_overrides, key, v)
                        i += 1
            i += 1
        
        if cli_overrides:
            args = cli_overrides
    
    # Merge with overrides
    if args:
        config = OmegaConf.merge(config, args)
    
    return config


def _nested_set(d, key, value):
    """Set nested dict value using dot notation. E.g., 'trainer.max_epochs' = 10."""
    keys = key.split('.')
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def instantiate_components(config):
    """Instantiate model, datamodule, and trainer from config.
    
    Args:
        config: OmegaConf config with model, data, trainer sections
    
    Returns:
        Tuple of (model, datamodule, trainer)
    """
    # Try to resolve all OmegaConf interpolations
    # Skip if env vars are missing (in notebook mode without ACCELERATOR set)
    try:
        OmegaConf.resolve(config)
    except Exception:
        # Fall back to partial resolution - convert to container which handles missing env vars
        pass
    
    # Convert to plain dicts (with resolved values where possible)
    model_config = OmegaConf.to_container(config.model)
    data_config = OmegaConf.to_container(config.data)
    trainer_config = OmegaConf.to_container(config.trainer)
    
    # Instantiate classes from their class_path + init_args
    model = instantiate_class_path(model_config)
    datamodule = instantiate_class_path(data_config)
    trainer = instantiate_class_path(trainer_config)
    
    return model, datamodule, trainer


def run_training_notebook(notebook_name, args=None):
    """Run training directly in notebook mode using registry-based config lookup.
    
    Args:
        notebook_name: Name of the training notebook (e.g., 'train_classifier')
                      Looks up configs/{notebook_name}.yaml and resolves model/data via train_classpaths.yaml
        args: Optional dict of training config overrides (e.g., {"trainer": {"max_epochs": 1}})
    """
    from pathlib import Path
    
    # Find training config path
    config_path = Path(f"configs/{notebook_name}.yaml")
    if not config_path.exists():
        config_path = Path("..") / f"configs/{notebook_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {notebook_name}.yaml")
    
    # Load training config
    config = OmegaConf.load(config_path)
    config_dir = config_path.parent
    
    # Load trainer config (referenced in training config)
    if isinstance(config.get('trainer'), str):
        config.trainer = OmegaConf.load(config_dir / config.trainer)
    
    # Load classpaths registry
    classpaths_path = config_dir / "train_classpaths.yaml"
    if not classpaths_path.exists():
        classpaths_path = Path("..") / "configs/train_classpaths.yaml"
    
    classpaths = OmegaConf.load(classpaths_path)
    
    # Strip .yaml extension to get registry keys
    # Config now has "mnist_litmodule.yaml" but registry has "mnist_litmodule"
    model_registry_key = config.model.replace('.yaml', '') if isinstance(config.model, str) else config.model
    data_registry_key = config.data.replace('.yaml', '') if isinstance(config.data, str) else config.data
    
    model_class_path = getattr(classpaths, model_registry_key)
    data_class_path = getattr(classpaths, data_registry_key)
    
    # Load component configs (using keys without .yaml)
    model_config_path = config_dir / f"{model_registry_key}.yaml"
    data_config_path = config_dir / f"{data_registry_key}.yaml"
    
    config.model = OmegaConf.load(model_config_path)
    config.data = OmegaConf.load(data_config_path)
    
    # Merge overrides
    if args:
        config = OmegaConf.merge(config, OmegaConf.create(args))
    
    # Resolve (skip on missing env vars)
    try:
        OmegaConf.resolve(config)
    except:
        pass
    
    # Convert to containers for instantiation
    model_args = OmegaConf.to_container(config.model)
    data_args = OmegaConf.to_container(config.data)
    trainer_args = OmegaConf.to_container(config.trainer)
    
    # Instantiate components
    model = instantiate_class_path({
        'class_path': model_class_path,
        'init_args': model_args
    })
    datamodule = instantiate_class_path({
        'class_path': data_class_path,
        'init_args': data_args
    })
    
    # Instantiate nested components in trainer args
    trainer_args = instantiate_class_path(trainer_args)
    trainer = Trainer(**trainer_args)
    
    # Train
    trainer.fit(model, datamodule=datamodule)


def run_training(args=None):
    """Run training in either CLI or notebook mode.
    
    Args:
        args: Optional dict of training config overrides (notebook mode).
              If None, uses CLI mode with sys.argv parsing via LightningCLI.
    """
    if args is None and not is_notebook():
        # CLI mode: Use LightningCLI directly
        from pytorch_lightning.cli import LightningCLI
        
        # Load train.yaml to extract model/data registry keys (with .yaml extension)
        train_config_path = Path("configs/train.yaml")
        if not train_config_path.exists():
            train_config_path = Path("..") / "configs/train.yaml"
        
        train_config = OmegaConf.load(train_config_path)
        config_dir = train_config_path.parent
        
        # Strip .yaml to get registry keys
        model_key = train_config.model.replace('.yaml', '') if isinstance(train_config.model, str) else None
        data_key = train_config.data.replace('.yaml', '') if isinstance(train_config.data, str) else None
        
        # Load classpaths registry
        classpaths_path = config_dir / "train_classpaths.yaml"
        if not classpaths_path.exists():
            classpaths_path = Path("..") / "configs/train_classpaths.yaml"
        
        classpaths = OmegaConf.load(classpaths_path)
        
        # Get actual class paths from registry
        model_class = _import_class(getattr(classpaths, model_key))
        data_class = _import_class(getattr(classpaths, data_key))
        
        # Use LightningCLI with parser_mode="omegaconf" to handle config resolution
        cli = LightningCLI(
            model_class,
            data_class,
            parser_kwargs={"parser_mode": "omegaconf"}
        )
    else:
        # Notebook mode: load configs, merge overrides, instantiate and train
        config = load_and_prepare_config(args)
        model, datamodule, trainer = instantiate_components(config)
        trainer.fit(model, datamodule=datamodule)
