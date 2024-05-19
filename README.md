<div align="center">
  <img src="https://img.shields.io/badge/language-Python-%233776AB.svg?logo=python">
  <img src="https://img.shields.io/badge/uses-Machine%20Learning-%232A2F3D.svg">
  <img src="https://custom-icon-badges.demolab.com/github/license/denvercoder1/custom-icon-badges?logo=law">
</div>

# <p align="center"> Breast Cancer Classification and Prediction <br> Using Machine Learning </p>

This program is designed to classify breast cancer using various machine learning models and analyze feature importance using SHAP (SHapley Additive exPlanations). The dataset used is the Breast Cancer Wisconsin Dataset.

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Modularity Split](#modularity-split)
- [Machine Learning Methods](#machine-learning-methods)
  - [Random Forest](#random-forest)
  - [Gradient Boosting](#gradient-boosting)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Neural Network](#neural-network)
  - [Logistic Regression](#logistic-regression)
- [Files](#files)
- [Example Output](#example-output)
- [License](#license)


## Features

- Prepares the Breast Cancer Wisconsin dataset by adding column names.
- Loads and preprocesses the data.
- Trains multiple machine learning models.
- Evaluates model performance.
- Performs SHAP analysis for feature importance.
- Compares the performance of different models.

## Installation

Ensure you have Python installed. Then, install the necessary libraries using pip:

```sh
pip install pandas scikit-learn matplotlib seaborn shap
```

## Dataset

Place the raw dataset file (`wdbc.data`) in the root directory of the project.

## Usage

Run the `main.py` script to execute the entire workflow:

```sh
python main.py
```

## Modularity Split

The program is divided into several modules for better organization and maintainability:

- `prepare_data.py`: Prepares the raw dataset by adding column names and saving it as a CSV file.
- `data_loader.py`: Loads the prepared data from the CSV file.
- `preprocessing.py`: Preprocesses the data by scaling features and splitting into training and test sets.
- `models.py`: Contains functions to retrieve various machine learning models.
- `evaluation.py`: Evaluates the models and plots confusion matrices.
- `shap_analysis.py`: Performs SHAP analysis and plots SHAP summary and dependence plots.
- `comparison.py`: Compares the performance of different models.
- `main.py`: The main script to execute the entire workflow.

## Machine Learning Methods

## Random Forest

**Overview** <br> Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes or mean prediction of the individual trees.

**Pros**:
- Handles large datasets with higher dimensionality.
- Reduces overfitting by averaging multiple decision trees.
- Can handle missing values and maintains accuracy for a large proportion of data.

**Cons**:
- Computationally expensive due to the creation of multiple trees.
- Less interpretable compared to single decision trees.

**Description** <br>
Random Forest algorithm combines multiple decision trees, each of which is trained on a random subset of the data. The final prediction is made by aggregating the predictions of all individual trees, typically by majority voting for classification tasks.

Let $T_1, T_2, \ldots, T_n$ be the $n$ decision trees in the forest. Each tree $T_i$ outputs a class prediction $h_i(x)$ for input $x$. The final prediction $\hat{y}$ is the mode of the class predictions:

$\hat{y} = \text{mode}(h_1(x), h_2(x), \ldots, h_n(x))$

For regression tasks, the final prediction is the average of the individual tree predictions.

## Gradient Boosting

**Overview** <br> Gradient Boosting is an ensemble technique that builds trees sequentially. Each new tree corrects errors made by the previous trees.

**Pros**:
- High prediction accuracy.
- Can handle a variety of data types (categorical, numerical).
- Provides feature importance.

**Cons**:
- Prone to overfitting if not properly tuned.
- Requires careful tuning of hyperparameters.

**Description** <br>
Gradient Boosting minimizes the loss function by adding weak learners using a gradient descent-like procedure. It iteratively adds decision trees to reduce the residual errors of previous trees.

Given a loss function $L(y, F(x))$, where $y$ is the true value and $F(x)$ is the model's prediction, the model is built in stages. At each stage $m$, a new tree $h_m(x)$ is added to the model to minimize the loss:

$F_{m}(x) = F_{m-1}(x) + \nu \cdot h_m(x)$

where $\nu$ is the learning rate, and $h_m(x)$ is the fitted tree at stage $m$.

## Support Vector Machine (SVM)

**Overview** <br>  SVM is a supervised learning algorithm that finds the hyperplane that best separates the classes in the feature space.

**Pros**:
- Effective in high-dimensional spaces.
- Works well for both linear and non-linear data.

**Cons**:
- Requires careful tuning of hyperparameters.
- Computationally intensive, especially with large datasets.

**Description** <br>
SVM aims to find the optimal hyperplane that maximizes the margin between the classes. The margin is defined as the distance between the hyperplane and the nearest points from each class, known as support vectors.

For a given dataset with input features $x_i$ and labels $y_i \in \{-1, 1\}$, the optimization problem is:

$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$
subject to $y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$,

where $\mathbf{w}$ is the weight vector, and $b$ is the bias term.

## Neural Network

**Overview** <br> Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data.

**Pros**:
- Capable of capturing complex patterns.
- Flexible architecture for various types of data.

**Cons**:
- Requires large amounts of data and computational power.
- Prone to overfitting and requires regularization.

**Description** <br>
Neural Networks consist of layers of interconnected nodes (neurons) where each connection represents a weighted path. The network learns by adjusting the weights through backpropagation to minimize the error.

For a given input $\mathbf{x}$, the output of a neuron in layer $l$ is:

$a_j^{(l)} = f\left( \sum_i w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)} \right)$

where $a_i^{(l-1)}$ is the activation from the previous layer, $w_{ji}^{(l)}$ is the weight from neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$, $b_j^{(l)}$ is the bias term, and $f$ is the activation function (e.g., sigmoid, ReLU).

The loss function $L$ (e.g., mean squared error for regression, cross-entropy for classification) is minimized using backpropagation, updating the weights $w$ and biases $b$ using gradient descent:

$w \leftarrow w - \eta \frac{\partial L}{\partial w}$
$b \leftarrow b - \eta \frac{\partial L}{\partial b}$

where $\eta$ is the learning rate.

## Logistic Regression

**Overview** <br>  Logistic Regression is a statistical model that predicts the probability of a binary outcome.

**Pros**:
- Simple and easy to implement.
- Provides probabilities and interpretable coefficients.

**Cons**:
- Assumes linearity between the independent variables and the log odds.
- Not suitable for complex relationships in data.

**Description** <br>
Logistic Regression models the probability of a binary outcome using the logistic function. It predicts the probability that a given input $\mathbf{x}$ belongs to the positive class (usually labeled as 1).

The logistic function (or sigmoid function) is defined as:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

For a given input $\mathbf{x}$, the model calculates the linear combination of the input features and weights:

$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$

where $\beta_0$ is the intercept (bias) and $\beta_1, \beta_2, \ldots, \beta_n$ are the model coefficients (weights). The logistic function is then applied to $z$ to obtain the probability of the positive class:

$P(Y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n)}}$

The model is trained by optimizing the log-likelihood function, which measures the fit of the model to the training data. The log-likelihood for logistic regression is given by:

$$\ell(\beta) = \sum_{i=1}^{m} \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$

where $y_i$ is the true label for the $i$-th training example, $\sigma(z_i)$ is the predicted probability for the $i$-th training example, and $m$ is the number of training examples.

The weights $\beta$ are updated using gradient descent to maximize the log-likelihood function:

$\beta \leftarrow \beta + \eta \nabla \ell(\beta)$

where $\eta$ is the learning rate and $\nabla \ell(\beta)$ is the gradient of the log-likelihood function with respect to the weights.

This process is repeated until convergence, resulting in a set of weights that best fit the training data.

## Files

- `prepare_data.py`
- `data_loader.py`
- `preprocessing.py`
- `models.py`
- `evaluation.py`
- `shap_analysis.py`
- `comparison.py`
- `main.py`

## Example Output

After running `main.py`, you will see the following outputs:
- Evaluation metrics for each model.
- Confusion matrices.
- SHAP summary plots.
- SHAP dependence plots.
- Comparison plots of model performance.

## License

This project is licensed under the MIT License.

