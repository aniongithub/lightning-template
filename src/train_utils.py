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


def instantiate_class_path(config):
    """Recursively instantiate classes from class_path + init_args."""
    if isinstance(config, dict):
        if 'class_path' in config:
            # This is a class to instantiate
            class_path = config['class_path']
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
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
        args: Optional dict of overrides
    
    Returns:
        OmegaConf config with all referenced files loaded
    """
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
    
    # Merge with overrides
    if args:
        config = OmegaConf.merge(config, args)
    
    return config


def instantiate_components(config):
    """Instantiate model, datamodule, and trainer from config.
    
    Args:
        config: OmegaConf config with model, data, trainer sections
    
    Returns:
        Tuple of (model, datamodule, trainer)
    """
    from src.datamodules.mnist_datamodule import MNISTDataModule
    from src.modules.mnist_litmodule import MNISTLitModule
    
    # Extract and instantiate components
    # Handle both wrapped (with init_args) and bare configs
    model_args = config.model.init_args if 'init_args' in config.model else config.model
    data_args = config.data.init_args if 'init_args' in config.data else config.data
    trainer_args = config.trainer.init_args if 'init_args' in config.trainer else config.trainer
    
    # Instantiate nested components (e.g., net inside model_args)
    model_args = instantiate_class_path(OmegaConf.to_container(model_args))
    data_args = instantiate_class_path(OmegaConf.to_container(data_args))
    trainer_args = instantiate_class_path(OmegaConf.to_container(trainer_args))
    
    model = MNISTLitModule(**model_args)
    datamodule = MNISTDataModule(**data_args)
    trainer = Trainer(**trainer_args)
    
    return model, datamodule, trainer


def run_training(args=None):
    """Run training in either CLI or notebook mode.
    
    Args:
        args: Optional dict of training config overrides (notebook mode).
              If None, uses CLI mode with sys.argv parsing via LightningCLI.
    """
    if args is None and not is_notebook():
        # CLI mode: LightningCLI parses sys.argv and config file
        from pytorch_lightning.cli import LightningCLI
        from src.datamodules.mnist_datamodule import MNISTDataModule
        from src.modules.mnist_litmodule import MNISTLitModule
        
        cli = LightningCLI(MNISTLitModule, MNISTDataModule, seed_everything=42)
    else:
        # Notebook mode: load configs, merge overrides, instantiate and train
        config = load_and_prepare_config(args)
        model, datamodule, trainer = instantiate_components(config)
        trainer.fit(model, datamodule=datamodule)
