from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model and print the results.

    :param model: sklearn.base.BaseEstimator
        The trained machine learning model.
    :param X_test: np.ndarray
        The test features.
    :param y_test: np.ndarray
        The test labels.
    :param model_name: str
        The name of the model.
    :return: tuple
        accuracy, roc_auc: The accuracy and ROC AUC score of the model.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Results for {model_name}:")
    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return accuracy, roc_auc


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot the confusion matrix for the model.

    :param model: sklearn.base.BaseEstimator
        The trained machine learning model.
    :param X_test: np.ndarray
        The test features.
    :param y_test: np.ndarray
        The test labels.
    :param model_name: str
        The name of the model.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
