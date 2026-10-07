import pandas as pd


def load_data(file_path="amazon.csv"):
    """Load the Amazon dataset."""
    return pd.read_csv(file_path)


def validate_columns(df, required_columns):
    """Check that required columns exist."""
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True


def top_rated_products(df, n=10):
    """Return the highest-rated products."""
    validate_columns(df, ["product_name", "rating", "rating_count"])

    data = df[
        ["product_name", "rating", "rating_count"]
    ].copy()

    data["rating"] = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    data["rating_count"] = (
        data["rating_count"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    data["rating_count"] = pd.to_numeric(
        data["rating_count"],
        errors="coerce",
    )

    result = (
        data
        .dropna(subset=["rating"])
        .sort_values(
            ["rating", "rating_count"],
            ascending=[False, False],
        )
        .head(n)
    )

    return result


def highest_discount_products(df, n=10):
    """Return products with the highest discount percentage."""
    validate_columns(
        df,
        [
            "product_name",
            "discount_percentage",
            "actual_price",
            "discounted_price",
        ],
    )

    data = df[
        [
            "product_name",
            "discount_percentage",
            "actual_price",
            "discounted_price",
        ]
    ].copy()

    data["discount_percentage"] = (
        data["discount_percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    data["discount_percentage"] = pd.to_numeric(
        data["discount_percentage"],
        errors="coerce",
    )

    result = (
        data
        .dropna(subset=["discount_percentage"])
        .sort_values(
            "discount_percentage",
            ascending=False,
        )
        .head(n)
    )

    return result


def category_product_counts(df):
    """Count products by category."""
    validate_columns(df, ["category"])

    result = (
        df["category"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    result.columns = ["category", "product_count"]

    return result


def average_rating_by_category(df):
    """Calculate average rating by category."""
    validate_columns(df, ["category", "rating"])

    data = df[["category", "rating"]].copy()

    data["rating"] = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    data = data.dropna(subset=["rating"])

    result = (
        data.groupby("category")["rating"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    result.columns = ["category", "average_rating"]

    return result


def most_reviewed_products(df, n=10):
    """Return products with the highest number of ratings/reviews."""
    validate_columns(
        df,
        ["product_name", "rating", "rating_count"],
    )

    data = df[
        ["product_name", "rating", "rating_count"]
    ].copy()

    data["rating"] = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    data["rating_count"] = (
        data["rating_count"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    data["rating_count"] = pd.to_numeric(
        data["rating_count"],
        errors="coerce",
    )

    result = (
        data
        .dropna(subset=["rating_count"])
        .sort_values(
            "rating_count",
            ascending=False,
        )
        .head(n)
    )

    return result


def highly_reviewed_low_rated_products(
    df,
    min_reviews=1000,
    max_rating=4.0,
    n=10,
):
    """Find products with many reviews but relatively low ratings."""
    validate_columns(
        df,
        ["product_name", "rating", "rating_count"],
    )

    data = df[
        ["product_name", "rating", "rating_count"]
    ].copy()

    data["rating"] = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    data["rating_count"] = (
        data["rating_count"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    data["rating_count"] = pd.to_numeric(
        data["rating_count"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["rating", "rating_count"]
    )

    result = (
        data[
            (data["rating_count"] >= min_reviews)
            & (data["rating"] <= max_rating)
        ]
        .sort_values(
            ["rating_count", "rating"],
            ascending=[False, True],
        )
        .head(n)
    )

    return result