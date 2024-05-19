import shap
import matplotlib.pyplot as plt


def shap_analysis(model, X_train, model_name):
    """
    Perform SHAP analysis and plot the results.

    :param model: sklearn.base.BaseEstimator
        The trained machine learning model.
    :param X_train: np.ndarray
        The training features.
    :param model_name: str
        The name of the model.
    """
    # Create an explainer for the model
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer(X_train)

    # Print shapes for debugging
    print(f"Shape of shap_values.values: {shap_values.values.shape}")
    print(f"Shape of X_train: {X_train.shape}")

    # Ensure the dimensions match
    assert shap_values.values.shape[1] == X_train.shape[
        1], "Mismatch in dimensions between SHAP values and input features"

    # Summary plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'SHAP Summary Plot for {model_name}')
    plt.show()

    # Dependence plot (example for the first feature)
    feature_index = 0  # Index of the feature to plot
    shap.dependence_plot(feature_index, shap_values[:, feature_index], X_train, show=False)
    plt.title(f'SHAP Dependence Plot for {model_name}')
    plt.show()
