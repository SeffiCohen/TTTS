# TTTS: Tree Test Time Simulation

Tree Test Time Simulation (TTTS) is an innovative methodology developed to enhance the robustness of decision trees against adversarial examples. The approach is based on the paper titled 'TTTS: Tree Test Time Simulation for Enhancing Decision Tree Robustness Against Adversarial Examples,' authored by Cohen Seffi, Arbili Ofir, Mirsky Yisroel, and Rokach Lior, and published in the proceedings of the AAAI 2024 conference. TTTS embodies the practical application of these research insights, offering a reliable and adaptable solution to improve the resistance of decision tree models to adversarial attacks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Introduction

Decision trees, while widely used for tabular data learning tasks, are vulnerable to adversarial attacks. TTTS introduces a novel inference-time methodology that integrates Monte Carlo simulations into decision trees, thereby significantly enhancing their robustness. This probabilistic approach to decision paths maintains the core structure of the tree while improving model performance and resilience to various adversarial attacks.

## Features

- **Robust Simulation**: Incorporates Monte Carlo methods into decision trees, introducing a probabilistic modification to decision paths without altering the tree structure.
- **Enhanced Performance**: Empirical analysis on 50 datasets shows improved model performance and robustness against white-box and black-box attacks.
- **Customizable Probability Types**: Supports various probability types to influence path selection during the prediction phase, enhancing decision-making under uncertainty.
- **Extensive Dataset Collection**: Comes with 50 datasets used for comprehensive empirical analysis and experimentation.
- **Comprehensive Documentation**: Includes detailed documentation, usage examples, and results of experiments.

## Installation

TTTS can be easily installed using pip:
```python
pip install TTTS
```

## Project Structure

The TTTS project is organized as follows to ensure ease of use, reproducibility of results, and a clear understanding of the project's components:

- `code/`: Contains the implementation of the TTTS algorithm.
  - `MonteCarloDecisionTreeClassifier.py`: The main class implementing the TTTS methodologies. This classifier introduces a probabilistic approach to decision-making in decision trees, using Monte Carlo simulations.
- `data/`: Includes the 50 datasets used for experiments. These datasets are utilized to demonstrate the effectiveness and robustness of the TTTS methodology against adversarial attacks.
- `experiments/`: Contains notebooks for running the experiments outlined in the paper. This folder allows users to reproduce the results and to understand the impact of TTTS on model performance and robustness.
- `paper/`: Includes a PDF file of the original research article. This provides users with direct access to the theoretical background, empirical analysis, and insights derived from the research.'


## Usage

After installing TTTS, you can seamlessly integrate it into your machine learning pipeline. The TTTS classifier enhances traditional decision tree classifiers by introducing robustness against adversarial attacks. Below is a straightforward code demonstration:

```python
from TTTS import MonteCarloDecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load the breast cancer dataset
bc = load_breast_cancer()
X, y = bc.data, bc.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Initialize the MonteCarloDecisionTreeClassifier with prob_type depth
clf = MonteCarloDecisionTreeClassifier(prob_type='depth')

# Train the classifier and make predictions
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Evaluate the classifier using AUC
auc = roc_auc_score(y_test, y_pred[:,1])
print(f'AUC: {auc}')
```

The prob_type parameter can be one of the following: ['fixed', 'depth', 'certainty', 'agreement', 'distance', 'confidence'].
  
## Experiments
Navigate to the experiments/ folder to view or run the experiments conducted as part of the research. This section includes scripts for reproducing the results presented in the paper, as well as detailed analysis of the TTTS performance under various conditions.

## Citation
@inproceedings{ttts2024,
  title={TTTS: Tree Test Time Simulation for Enhancing Decision Tree Robustness Against Adversarial Examples},
  author={Cohen Seffi , Arbili Ofir , Mirsky Yisroel , Rokach Lior},
  booktitle={Proceedings of the AAAI Conference},
  year={2024}
}

## License
This project is licensed under the MIT License.

## Contact

