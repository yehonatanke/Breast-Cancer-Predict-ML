import matplotlib.pyplot as plt


def compare_models(results):
    """
    Compare the performance of different models.

    :param results: list
        A list of dictionaries containing model names and their performance metrics.
    """
    model_names = [result['model_name'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    roc_aucs = [result['roc_auc'] for result in results]

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()

    # Plot ROC AUC comparison
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, roc_aucs, color='skyblue')
    plt.xlabel('ROC AUC Score')
    plt.title('Model ROC AUC Comparison')
    plt.show()
