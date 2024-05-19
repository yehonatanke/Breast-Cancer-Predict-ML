from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def get_model(model_name):
    """
    Get a machine learning model based on the model name.

    :param model_name: str
        The name of the machine learning model to retrieve.
    :return: sklearn.base.BaseEstimator
        The machine learning model instance.
    :raises ValueError: If the model name is not recognized.
    """
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "neural_network": MLPClassifier(random_state=42),
        "logistic_regression": LogisticRegression(random_state=42)
    }

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not recognized. Choose from {list(models.keys())}.")

    return models[model_name]
