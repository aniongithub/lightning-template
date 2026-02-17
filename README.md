# Lightning Template

A notebook-first PyTorch Lightning template powered by nbdev for automatic code generation. Edit notebooks in `nbs/`, and source code and config files are automatically generated. 

This example trains a SimpleDenseNet on MNIST, demonstrating clean separation between configuration, data loading, model architecture, and training logic—all organized as readable, testable notebooks.

## Getting Started

### 1. Setup
- Install Docker and Visual Studio Code
- Install the Remote Development extension pack
- Open the project folder in a dev container (`Shift+Ctrl+P` → "Dev Containers: Open Folder in Container")

See [here](https://code.visualstudio.com/docs/devcontainers/tutorial) for more detailed instructions for your OS/platform

### 2. Build & Generate Source Code
Press `Shift+Ctrl+B` (or run the "Build & Hydrate" task) to regenerate Python source files from notebooks via nbdev. Do this whenever you modify any notebook.

### 3. Run Training

**Option A: Notebook Mode (Interactive Development)**

Open [nbs/train.ipynb](nbs/train.ipynb) and run:
- **Single Dev Run**: 1 epoch, 10 batches per split → validates entire pipeline in seconds
- **Full Training Run**: 10 epochs, all data → trains complete model

Metrics are automatically logged to TensorBoard during training.

Note: When you make changes to notebooks, you might need to restart your Jupyter kernel to reload the changes python files.

**Option B: CLI Mode (Production)**

Use the VS Code launch configuration: Press `F5` or go to Run → Start Debugging, then select a training launch config. Or run manually:

```bash
python src/train.py fit --config configs/train.yaml
```

See `.vscode/launch.json` for pre-configured training commands.

### 4. Monitor Training with TensorBoard

During or after training, launch TensorBoard via VS Code command palette:
- `Ctrl+Shift+P` → "Python: Launch TensorBoard" 
- Point to `./logs` directory (or, during interactive development to `./nbs/logs`)
- Dashboard displays loss, accuracy, and best validation metrics in real-time

## Project Structure

```
src/
  train.py                    # Auto-generated CLI entrypoint from train.ipynb
  train_utils.py              # Shared training utilities for CLI & notebook modes
  lightning_config.py         # Config decorators for notebook-to-Python export
  datamodules/
    mnist_datamodule.py       # Auto-generated from nbs/mnist_datamodule.ipynb
  modules/
    mnist_litmodule.py        # Auto-generated from nbs/mnist_litmodule.ipynb
  models/
    simpledense.py            # Auto-generated from nbs/simpledense.ipynb

nbs/                          # Editable notebooks (source of truth)
  train.ipynb                 # Training entrypoint with dev & full-training cells
  mnist_datamodule.ipynb      # Data loading & preprocessing with tests
  mnist_litmodule.ipynb       # PyTorch Lightning training logic
  simpledense.ipynb           # Model architecture definition

configs/
  train.yaml                  # Training config (model + data + trainer)
  trainer/default.yaml        # Trainer settings (epochs, callbacks, logging)
  mnist_datamodule.yaml       # DataModule hyperparams
  mnist_litmodule.yaml        # LitModule hyperparams
```

## Key Design Patterns

**Notebook-First Development**: Edit notebooks in `nbs/`, not Python files in `src/` - these will be overwritten for each generation. Run "Build & Hydrate" (`Shift+Ctrl+B`) to generate source code automatically via nbdev. Cells marked with `#| export` are exported while others are for testing/exploration.

**Automatic Config Generation**: Decorators like `@lightning_config`, `@training_config`, and `@model` automatically generate YAML config stubs from function signatures. Edit the generated YAML files to configure CLI run behavior, with no manual Python editing needed.

**Decorators for Organization**: Use `@patch` to add methods to classes across multiple notebook cells, keeping each cell focused and well-documented. This beats giant monolithic cells with full class definitions inside them, while still conforming to Pytorch Lightning paradigms.

**Separate CLI & Notebook Modes**: 
- `run_training()` uses LightningCLI for production CLI
- `run_training_notebook()` directly instantiates components for interactive work

**Config System**: Unified OmegaConf tree merges YAML configs, resolves references, then extracts only needed parts. Single source of truth across CLI and notebook modes.

**Component Organization**: Data, model, and training logic are separate, testable modules with clear interfaces. Each has a corresponding notebook with documentation and test cells.
