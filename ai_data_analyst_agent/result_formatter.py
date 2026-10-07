import pandas as pd


def format_analysis_result(result):
    """
    Convert a pandas DataFrame into a JSON-friendly
    representation for the AI layer.
    """

    if result is None:
        return {
            "row_count": 0,
            "columns": [],
            "rows": [],
        }

    if not isinstance(result, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    clean_result = result.copy()

    clean_result = clean_result.astype(object).where(
        pd.notnull(clean_result),
        None,
    )

    return {
        "row_count": len(clean_result),
        "columns": clean_result.columns.tolist(),
        "rows": clean_result.to_dict(orient="records"),
    }