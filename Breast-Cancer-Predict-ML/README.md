<div align="center">
  <img src="https://img.shields.io/badge/language-Python-%233776AB.svg?logo=python">
  <img src="https://img.shields.io/badge/uses-Machine%20Learning-%232A2F3D.svg">
  <img src="https://custom-icon-badges.demolab.com/github/license/denvercoder1/custom-icon-badges?logo=law">
</div>

# <p align="center"> Breast Cancer Classification and Prediction <br> Using Machine Learning </p>

This program is designed to classify breast cancer using various machine learning models and analyze feature importance using SHAP (SHapley Additive exPlanations). The dataset used is the Breast Cancer Wisconsin Dataset.

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

### 1. Random Forest

**Description**: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes or mean prediction of the individual trees.

**Pros**:
- Handles large datasets with higher dimensionality.
- Reduces overfitting by averaging multiple decision trees.
- Can handle missing values and maintains accuracy for a large proportion of data.

**Cons**:
- Computationally expensive due to the creation of multiple trees.
- Less interpretable compared to single decision trees.

**Mathematical Description**:
Random Forest algorithm combines multiple decision trees, each of which is trained on a random subset of the data. The final prediction is made by aggregating the predictions of all individual trees, typically by majority voting for classification tasks.

### 2. Gradient Boosting

**Description**: Gradient Boosting is an ensemble technique that builds trees sequentially. Each new tree corrects errors made by the previous trees.

**Pros**:
- High prediction accuracy.
- Can handle a variety of data types (categorical, numerical).
- Provides feature importance.

**Cons**:
- Prone to overfitting if not properly tuned.
- Requires careful tuning of hyperparameters.

**Mathematical Description**:
Gradient Boosting minimizes the loss function by adding weak learners using a gradient descent-like procedure. It iteratively adds decision trees to reduce the residual errors of previous trees.

### 3. Support Vector Machine (SVM)

**Description**: SVM is a supervised learning algorithm that finds the hyperplane that best separates the classes in the feature space.

**Pros**:
- Effective in high-dimensional spaces.
- Works well for both linear and non-linear data.

**Cons**:
- Requires careful tuning of hyperparameters.
- Computationally intensive, especially with large datasets.

**Mathematical Description**:
SVM aims to find the optimal hyperplane that maximizes the margin between the classes. It uses kernel functions to transform the data into a higher-dimensional space where it becomes linearly separable.

### 4. Neural Network

**Description**: Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data.

**Pros**:
- Capable of capturing complex patterns.
- Flexible architecture for various types of data.

**Cons**:
- Requires large amounts of data and computational power.
- Prone to overfitting and requires regularization.

**Mathematical Description**:
Neural Networks consist of layers of interconnected nodes where each connection represents a weighted path. The network learns by adjusting the weights through backpropagation to minimize the error.

### 5. Logistic Regression

**Description**: Logistic Regression is a statistical model that predicts the probability of a binary outcome.

**Pros**:
- Simple and easy to implement.
- Provides probabilities and interpretable coefficients.

**Cons**:
- Assumes linearity between the independent variables and the log odds.
- Not suitable for complex relationships in data.

**Mathematical Description**:
Logistic Regression models the probability of the default class (usually 0 or 1) using the logistic function:
$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}}$

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

