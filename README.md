# Supporting Materials for "A Multitask Dynamically Interconnected Learning Approach for Physics-Informed Neural Networks"

This repository contains the supporting materials for the manuscript titled **"A Multitask Dynamically Interconnected Learning Approach for Physics-Informed Neural Networks"**. The materials include the implementation of a **Physics-Informed Neural Network (PINN)** designed to solve **partial differential equations (PDEs)** with multitask learning capabilities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [License](#license)

## Overview
The **PINN model** implemented in this repository is tailored for solving PDEs with boundary and initial conditions using a **multitask dynamically interconnected learning approach**. This approach integrates multiple physical laws into the neural network architecture, enabling it to learn solutions that satisfy both data and governing equations.

### Key aspects of the model include:
- **Neural Network Architecture**: A **multi-layer perceptron (MLP)** is used to approximate the solution of the PDE.
- **Loss Function**: The loss function combines **data-driven terms** and **physics-based residual terms** to ensure the solution adheres to both observed data and physical constraints.
- **Optimization**: The model uses the **Adam optimizer** and **L-BFGS-B algorithm** for training.

## Features
- Solves PDEs with both **data-driven** and **physics-informed** approaches.
- Supports **custom initialization** of neural network weights and biases.
- Implements **advanced loss terms** for better convergence and accuracy.
- Includes support for **saving and loading trained models**.

## Requirements
To run this code, you will need the following dependencies:

- **Python** 3.6
- **TensorFlow** v1.14.0-rc1
- **NumPy**
- **SciPy**

You can install the required packages using pip:

```bash
pip install tensorflow==v1.14.0-rc1 
```

## Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/zhangxiao726/MDI_PINNs.git
cd pinn
```
## Usage
To train the PINN model, execute the following command:
```bash
python main.py
```
The script will initialize the model, load the data, and start the training process. Training progress will be logged to the console.After training, you can use the predict method to generate predictions for new input data. 


## Training Parameters and Data Sampling

### Training Parameters

- **Total iterations**: 2,200  
- **Random seeds**:  
  - NumPy: `1234`  
  - TensorFlow: `42`  
- **Optimizer**: L-BFGS-B  
  - `'maxiter'`: 7000  
  - `'maxfun'`: 20000  
  - `'maxcor'`: 70  
  - `'maxls'`: 70  

### Data Sampling Strategy

The sampling strategy for various domains and regions is summarized below. For full implementation details, see the file: [main.py](https://github.com/zhangxiao726/MDI_PINNs/blob/main/main.py).

| Parameter | Value | Spatial Domain | Sampling Method     |
|----------|-------|----------------|----------------------|
| N0       | 400   | x ∈ [0, 0.11]  | Random selection     |
|          | 400   | x ∈ [-0.11, 0] | Random selection     |
| N_b      | 2,000 | t ∈ [0, 800]   | Random selection     |
| N_f_1    | 4,500 | Full domain    | Latin Hypercube      |
| N_f_2    | 4,600 | Central region | Latin Hypercube      |

> **Note**:  
> - `N0`: Number of initial condition points.  
> - `N_b`: Number of boundary condition points.  
> - `N_f_1`, `N_f_2`: Collocation points used to enforce PDE residuals over different spatial subdomains.

### Domain Configuration

The computational domain is divided into distinct regions to facilitate multitask learning and dynamic interconnection between subproblems.

| Region        | Spatial Range       | Temporal Range     |
|---------------|---------------------|--------------------|
| u-space       | [0.0, 0.11]         | [0.0, 800.0]       |
| v-space       | [-0.11, 0.0]        | [0.0, 800.0]       |
| u-central     | [0.0, 0.05]         | [0.0, 800.0]       |
| v-central     | [-0.05, 0.0]        | [0.0, 800.0]       |


**Example usage:**
```bash
# Load the trained model
model = PhysicsInformedNN(...)

# Prepare input data
X_u_star = ...
X_v_star = ...

# Generate predictions
u_star, v_star, f_u_star, f_v_star, D_star, D1_star = model.predict(X_u_star, X_v_star)
```

## Code Structure
The main components of the code are organized as follows:

- **`PhysicsInformedNN.py`**: Contains the definition of the `PhysicsInformedNN` class, which implements the PINN model.
- **`main.py`**: Entry point for running the model. Handles data loading, model initialization, and training.
- **`data/`**: Directory for storing input data and results.
- **`results/`**: Directory for saving trained models and prediction outputs.

## Results
The model generates predictions for the solution of the PDE, including:

- **`u_star`**: Solution for variable **u**.
- **`v_star`**: Solution for variable **v**.
- **`f_u_star, f_v_star`**: Residuals for variables **u** and **v**.
- **`D_star, D1_star`**: Diffusion coefficients.

Training progress is logged to files **`the_c.txt`** and **`the_c_x.txt`** for further analysis.

## License
This project is licensed under the **MIT License**. 
