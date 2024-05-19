from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(df, target_column):
    """
    Preprocess data by scaling features and splitting into training and test sets.

    :param df: pd.DataFrame
        The input data as a pandas DataFrame.
    :param target_column: str
        The name of the target column.
    :return: tuple
        X_train, X_test, y_train, y_test: The training and testing sets.
    :raises ValueError: If the target column is not in the DataFrame.
    """
    if target_column not in df.columns:
        raise ValueError(f"The target column '{target_column}' is not in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
