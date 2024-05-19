import os
from data_loader import load_data
from preprocessing import preprocess_data
from models import get_model
from evaluation import evaluate_model, plot_confusion_matrix
from shap_analysis import shap_analysis
from comparison import compare_models
from prepare_data import prepare_data


def main():
    """
    Main function to prepare data, load data, preprocess, train models, evaluate, perform SHAP analysis, and compare results.
    """
    input_file_path = 'wdbc.data'
    output_file_path = 'breast_cancer_wisconsin.csv'  # The prepared CSV file

    # Prepare data
    if not os.path.exists(output_file_path):
        prepare_data(input_file_path, output_file_path)

    # Load data
    data = load_data(output_file_path)

    # Preprocess data
    target_column = 'diagnosis'  # The target column name
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)

    # Define models to train
    model_names = ['random_forest', 'gradient_boosting', 'svm', 'neural_network', 'logistic_regression']

    # Store results for comparison
    results = []

    for model_name in model_names:
        model = get_model(model_name)
        model.fit(X_train, y_train)

        # Evaluate model
        accuracy, roc_auc = evaluate_model(model, X_test, y_test, model_name)
        plot_confusion_matrix(model, X_test, y_test, model_name)

        # SHAP analysis
        shap_analysis(model, X_train, model_name)

        # Store results
        results.append({
            'model_name': model_name,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        })

    # Compare models
    compare_models(results)


if __name__ == '__main__':
    main()
