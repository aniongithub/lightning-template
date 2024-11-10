# Lightning Template

A devcontainer-based Pytorch Lightning template project for CPU/GPU training. The sample project is a DenseNet that is trained on the MNIST handwritten digits dataset.

## Getting started

1. **Install Docker**: Ensure you have Docker installed on your machine. You can download it from [Dockers official website](https://www.docker.com/products/docker-desktop).
2. **Install Visual Studio Code**: Download and install Visual Studio Code from [here](https://code.visualstudio.com/).
3. **Install Remote Development Extension Pack**: In Visual Studio Code, install the Remote Development extension pack. You can find it in the Extensions view by searching for `Remote Development`.
4. **Configure `.devcontainer/devcontainer.json`**: Based on what accelerator you want to use (CPU/GPU), uncomment the appropriate Dockerfile. If you chose GPU, also uncomment `runArgs`

   ```json
   {
   	"name": "${localWorkspaceFolderBasename}",
   	"build": {
   		"context": "..",

   		// Use the appropriate Dockerfile for your environment
   		"dockerfile": "cpu.Dockerfile" // <-- For CPU
   		// "dockerfile": "gpu.Dockerfile" // <-- For GPU
   	},
   	"customizations": {
   		"vscode": {
   			"extensions": [
   				"cweijan.vscode-typora",
   				"ms-python.python",
   				"ms-toolsai.tensorboard"
   			]
   		}
   	},
   	"updateContentCommand": "pip3 install -r ${containerWorkspaceFolder}/requirements.txt",

   	// Uncomment runArgs when using GPU
   	// "runArgs": [
   	// 	"--privileged",
   	// 	"--gpus", "all"
   	// ]
   }
   ```
5. **Open the Project in a Dev Container**: Open your project in Visual Studio Code. Press `Shift+Ctrl+P` and select `Remote-Containers: Open Folder in Container...`. Choose your project folder.
6. **Verify the Setup**: Once the container is built and running, verify that your environment is set up correctly by running any of the available launch configurations `Train CLI` or `Train Dev Run`.
