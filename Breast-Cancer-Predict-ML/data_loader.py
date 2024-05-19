import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file.

    :param file_path: str
        The path to the CSV file.
    :return: pd.DataFrame
        The loaded data as a pandas DataFrame.
    :raises FileNotFoundError: If the file does not exist.
    :raises pd.errors.EmptyDataError: If the file is empty.
    :raises pd.errors.ParserError: If the file is malformed.
    """
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The provided CSV file is empty.")
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {file_path} does not exist.") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError("The provided CSV file is empty.") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError("The provided CSV file is malformed.") from e
