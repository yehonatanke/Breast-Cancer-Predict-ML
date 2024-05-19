import pandas as pd


def prepare_data(input_file_path, output_file_path):
    """
    Prepare the Breast Cancer Wisconsin dataset by adding column names and saving it as a CSV.

    :param input_file_path: str
        The path to the input file.
    :param output_file_path: str
        The path to save the prepared CSV file.
    :raises FileNotFoundError: If the input file does not exist.
    :raises pd.errors.EmptyDataError: If the input file is empty.
    :raises pd.errors.ParserError: If the input file is malformed.
    """
    column_names = [
        "ID", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean",
        "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    try:
        data = pd.read_csv(input_file_path, header=None, names=column_names)
        if data.empty:
            raise ValueError("The provided CSV file is empty.")
        data.to_csv(output_file_path, index=False)
        print(f"Data preparation complete. The data is saved as '{output_file_path}'.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {input_file_path} does not exist.") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError("The provided CSV file is empty.") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError("The provided CSV file is malformed.") from e


if __name__ == "__main__":
    prepare_data('wdbc.data', 'breast_cancer_wisconsin.csv')
