# Breast Cancer Classification and Prediction Using Machine Learning 

Classify breast cancer using machine learning models and analyze feature importance using SHAP. The dataset used is the Breast Cancer Wisconsin Dataset.

## Features

- Prepares the Breast Cancer Wisconsin dataset by adding column names.
- Loads and preprocesses the data.
- Trains multiple machine learning models.
- Evaluates model performance.
- Performs SHAP analysis for feature importance.
- Compares the performance of different models.

## Dataset

Place the raw dataset file (`wdbc.data`) in the root directory of the project.

## Moduls Split

- `prepare_data.py`: Prepares the raw dataset by adding column names and saving it as a CSV file.
- `data_loader.py`: Loads the prepared data from the CSV file.
- `preprocessing.py`: Preprocesses the data by scaling features and splitting into training and test sets.
- `models.py`: Contains functions to retrieve various machine learning models.
- `evaluation.py`: Evaluates the models and plots confusion matrices.
- `shap_analysis.py`: Performs SHAP analysis and plots SHAP summary and dependence plots.
- `comparison.py`: Compares the performance of different models.
- `main.py`: The main script to execute the entire workflow.

## Machine Learning Methods

* Random Forest
* Gradient Boosting
* SVM
* DNN
* Logistic Regression

## Example Output

After running the main file, you will see the following outputs:
- Evaluation metrics for each model.
- Confusion matrices.
- SHAP summary plots.
- SHAP dependence plots.
- Comparison plots of model performance.

