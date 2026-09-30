# ============================================================
# WEEK 2 - END-TO-END MACHINE LEARNING PROJECT
# CUSTOMER CHURN PREDICTION
#
# ONE-FILE PROJECT
# ----------------
# Run with Python:
#     python "customer churn.py"
#
# Run the Streamlit application:
#     streamlit run "customer churn.py"
#
# The same file handles both modes.
#
# The training mode creates:
#   - cleaned dataset
#   - EDA charts
#   - five written EDA insights
#   - Day 6 problem statement
#   - Day 7 EDA notebook
#   - Day 8 feature-engineering notebook
#   - model comparison
#   - evaluation summary
#   - test predictions
#   - feature importance
#   - trained model
#   - model metadata
#   - requirements.txt
#   - README.md
#   - .gitignore
#
# The Streamlit mode loads the saved model and provides the
# final customer churn prediction interface.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
from io import StringIO
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FOLDER = PROJECT_ROOT / "data"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

DATASET_PATH = DATA_FOLDER / "churn IBM dataset.csv"

MODEL_PATH = OUTPUT_FOLDER / "best_churn_model.joblib"
METADATA_PATH = OUTPUT_FOLDER / "model_metadata.json"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]


REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
]


# These columns are not legitimate predictive inputs because
# they are identifiers, targets, or outcome-related information.
LEAKAGE_COLUMNS = [
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "CLTV",
]


# Candidate predictors before constant-column checking.
CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "TenureGroup",
    "IsNewCustomer",
]


NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "AvgMonthlySpend",
    "TotalServices",
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================


def print_section(title):
    """Print a readable terminal section heading."""

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)



def clean_column_names(df):
    """Remove accidental spaces/BOM characters from column names."""

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


# ============================================================
# 5. LOAD THE EXACT DATASET
# ============================================================


def find_header_line(file_path):
    """
    Find the real dataset header.

    This protects the project from blank lines or copied text
    before the header.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:

        lines = file.readlines()

    for index, line in enumerate(lines):

        normalized = line.strip().replace(
            "\\t",
            "\t",
        )

        if (
            "CustomerID" in normalized
            and "Tenure Months" in normalized
            and "Churn Label" in normalized
        ):
            return index

    return None



def detect_separator(header_line):
    """Detect tab, comma, or semicolon separated data."""

    if "\\t" in header_line:
        return "\\t"

    if "\t" in header_line:
        return "\t"

    if "," in header_line:
        return ","

    if ";" in header_line:
        return ";"

    return None



def load_dataset():
    """
    Load the project dataset from the exact required filename:
        data/churn IBM dataset.csv

    The file is allowed to be a tab-separated file even though
    its extension is .csv.
    """

    print_section("LOADING CUSTOMER CHURN DATASET")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "\nDataset not found.\n\n"
            f"Expected exact path:\n{DATASET_PATH}\n\n"
            "Make sure the file is named exactly:\n"
            "churn IBM dataset.csv\n\n"
            "and is inside the project's data folder."
        )

    print(f"Dataset file: {DATASET_PATH}")

    header_index = find_header_line(DATASET_PATH)

    if header_index is None:

        raise ValueError(
            "\nThe dataset file was found, but its expected header "
            "could not be identified.\n\n"
            "Expected columns include CustomerID, Tenure Months, "
            "Monthly Charges, Total Charges, and Churn Label."
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:

        lines = file.readlines()

    data_text = "".join(
        lines[header_index:]
    )

    data_text = data_text.replace(
        "\\t",
        "\t",
    )

    header_line = data_text.splitlines()[0]

    separator = detect_separator(
        header_line
    )

    print(
        f"Header found at line: {header_index + 1}"
    )

    print(
        f"Detected separator: {repr(separator)}"
    )

    if separator == "\t":

        df = pd.read_csv(
            StringIO(data_text),
            sep="\t",
            skip_blank_lines=True,
            engine="python",
        )

    elif separator == ",":

        df = pd.read_csv(
            StringIO(data_text),
            sep=",",
            skip_blank_lines=True,
            engine="python",
        )

    elif separator == ";":

        df = pd.read_csv(
            StringIO(data_text),
            sep=";",
            skip_blank_lines=True,
            engine="python",
        )

    else:

        df = pd.read_csv(
            StringIO(data_text),
            sep=None,
            skip_blank_lines=True,
            engine="python",
        )

    df = clean_column_names(df)

    for column in df.select_dtypes(
        include="object"
    ).columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    print(
        f"Rows loaded: {df.shape[0]}"
    )

    print(
        f"Columns loaded: {df.shape[1]}"
    )

    return df


# ============================================================
# 6. DATASET VALIDATION
# ============================================================


def validate_dataset(df):
    """Check that the selected file is the expected churn dataset."""

    print_section("DATASET VALIDATION")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe dataset is missing these required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_columns
            )
        )

    print("Dataset validation successful.")

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if unexpected_columns:

        print(
            "\nAdditional columns found:"
        )

        for column in unexpected_columns:
            print(
                f" - {column}"
            )


# ============================================================
# 7. DATA CLEANING
# ============================================================


def clean_data(df):
    """Clean strings, numeric columns, duplicates, and target values."""

    print_section("DATA CLEANING")

    df = df.copy()

    # Clean object/string columns.
    for column in df.select_dtypes(
        include="object"
    ).columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Convert numerical columns.
    numeric_columns = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Value",
        "Churn Score",
        "CLTV",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Record missing-value counts before imputation.
    missing_before = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing_before = missing_before[
        missing_before > 0
    ]

    missing_before.to_csv(
        OUTPUT_FOLDER
        / "missing_values_before_modeling.csv",
        header=["MissingValues"],
    )

    # Remove duplicate customer records.
    rows_before = len(df)

    df = df.drop_duplicates(
        subset="CustomerID"
    ).copy()

    duplicates_removed = (
        rows_before - len(df)
    )

    print(
        f"Duplicate customers removed: {duplicates_removed}"
    )

    # Keep only usable target rows.
    df = df.dropna(
        subset=["Churn Label"]
    ).copy()

    df = df[
        df["Churn Label"].isin(
            ["Yes", "No"]
        )
    ].copy()

    # Binary machine-learning target.
    df["Churn"] = (
        df["Churn Label"]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
        .astype(int)
    )

    print(
        "Missing values in key numerical columns:"
    )

    print(
        df[
            [
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
            ]
        ].isnull().sum()
    )

    print(
        f"Rows after cleaning: {len(df)}"
    )

    return df


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================


def create_features(df):
    """Create features using information available at prediction time."""

    print_section("FEATURE ENGINEERING")

    df = df.copy()

    # --------------------------------------------------------
    # Average monthly spend
    # --------------------------------------------------------

    df["AvgMonthlySpend"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"]
        / df["Tenure Months"],
        df["Monthly Charges"],
    )

    # --------------------------------------------------------
    # Tenure group
    # --------------------------------------------------------

    df["TenureGroup"] = pd.cut(
        df["Tenure Months"],
        bins=[
            -1,
            6,
            12,
            24,
            48,
            np.inf,
        ],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49+ months",
        ],
    )

    # --------------------------------------------------------
    # New customer flag
    # --------------------------------------------------------

    df["IsNewCustomer"] = np.where(
        df["Tenure Months"] <= 6,
        "Yes",
        "No",
    )

    # --------------------------------------------------------
    # Number of subscribed services
    # --------------------------------------------------------

    service_columns = [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    def count_yes(row):

        total = 0

        for value in row:

            if (
                str(value)
                .strip()
                .lower()
                == "yes"
            ):

                total += 1

        return total

    df["TotalServices"] = (
        df[
            service_columns
        ].apply(
            count_yes,
            axis=1,
        )
    )

    print(
        "Created features:"
    )

    print(
        " - AvgMonthlySpend"
    )

    print(
        " - TenureGroup"
    )

    print(
        " - IsNewCustomer"
    )

    print(
        " - TotalServices"
    )

    return df


# ============================================================
# 9. DATA QUALITY AND FEATURE REPORT
# ============================================================


def save_data_quality_report(df):
    """Save a simple column-level quality report."""

    rows = []

    for column in df.columns:

        rows.append(
            {
                "Column": column,
                "DataType": str(df[column].dtype),
                "MissingValues": int(df[column].isna().sum()),
                "UniqueValues": int(df[column].nunique(dropna=True)),
            }
        )

    report = pd.DataFrame(rows)

    report.to_csv(
        OUTPUT_FOLDER
        / "data_quality_report.csv",
        index=False,
    )



def save_feature_engineering_notes(
    df,
    categorical_features,
    numerical_features,
    constant_features,
):
    """Write Day 8 feature-engineering documentation."""

    path = (
        OUTPUT_FOLDER
        / "feature_engineering_notes.md"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Day 8 - Feature Engineering\n\n"
        )

        file.write(
            "## Engineered features\n\n"
        )

        file.write(
            "- **AvgMonthlySpend:** Total Charges divided by Tenure Months when tenure is greater than zero; otherwise Monthly Charges is used.\n"
        )

        file.write(
            "- **TenureGroup:** Groups tenure into 0-6, 7-12, 13-24, 25-48, and 49+ months.\n"
        )

        file.write(
            "- **IsNewCustomer:** Yes when Tenure Months is 6 or fewer.\n"
        )

        file.write(
            "- **TotalServices:** Counts service fields containing Yes.\n\n"
        )

        file.write(
            "## Categorical preprocessing\n\n"
        )

        file.write(
            "Categorical missing values are filled with the most frequent category and then one-hot encoded. Unknown categories are ignored so the application can fail safely instead of crashing.\n\n"
        )

        file.write(
            "## Numerical preprocessing\n\n"
        )

        file.write(
            "Numerical missing values use median imputation. Numerical variables are standardized for the Logistic Regression pipeline.\n\n"
        )

        file.write(
            "## Leakage prevention\n\n"
        )

        file.write(
            "The following columns are excluded from prediction: "
            + ", ".join(LEAKAGE_COLUMNS)
            + ". They include identifiers, the target, or fields that can encode outcome information.\n\n"
        )

        file.write(
            "## Final model features\n\n"
        )

        file.write(
            "### Categorical\n\n"
        )

        for column in categorical_features:

            file.write(
                f"- {column}\n"
            )

        file.write(
            "\n### Numerical\n\n"
        )

        for column in numerical_features:

            file.write(
                f"- {column}\n"
            )

        if constant_features:

            file.write(
                "\n## Constant features in this local dataset\n\n"
            )

            file.write(
                "These fields had only one observed value in the supplied dataset and were excluded from the final model because they provide no variation during training:\n\n"
            )

            for column in constant_features:

                file.write(
                    f"- {column}\n"
                )


# ============================================================
# 10. EDA
# ============================================================


def calculate_eda_statistics(df):
    """Calculate the five required churn-pattern analyses."""

    contract_rates = (
        df.groupby(
            "Contract",
            observed=True,
        )["Churn"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    internet_rates = (
        df.groupby(
            "Internet Service",
            observed=True,
        )["Churn"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    payment_rates = (
        df.groupby(
            "Payment Method",
            observed=True,
        )["Churn"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    senior_rates = (
        df.groupby(
            "Senior Citizen",
            observed=True,
        )["Churn"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    tenure_rates = (
        df.groupby(
            "TenureGroup",
            observed=True,
        )["Churn"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    return (
        contract_rates,
        internet_rates,
        payment_rates,
        senior_rates,
        tenure_rates,
    )



def write_eda_insights(
    df,
    contract_rates,
    internet_rates,
    payment_rates,
    senior_rates,
    tenure_rates,
):
    """Create five written insights rather than only raw tables."""

    churn_rate = (
        df["Churn"].mean() * 100
    )

    insights = []

    # Insight 1 - contract coverage
    contract_text = (
        contract_rates.index.tolist()
    )

    if len(contract_rates) == 1:

        only_contract = contract_rates.index[0]

        only_rate = (
            contract_rates.iloc[0]["mean"]
            * 100
        )

        insights.append(
            f"1. Contract coverage is limited in this local dataset: only '{only_contract}' is represented, with a churn rate of {only_rate:.2f}%. Because no other contract type appears, this file cannot support a reliable between-contract churn comparison."
        )

    else:

        highest_contract = contract_rates.index[0]
        highest_contract_rate = (
            contract_rates.iloc[0]["mean"]
            * 100
        )

        insights.append(
            f"1. Contract pattern: '{highest_contract}' has the highest observed churn rate at {highest_contract_rate:.2f}% in this dataset. This is a descriptive association, not proof that contract type causes churn."
        )

    # Insight 2 - internet coverage
    if len(internet_rates) == 1:

        only_internet = internet_rates.index[0]

        only_rate = (
            internet_rates.iloc[0]["mean"]
            * 100
        )

        insights.append(
            f"2. Internet-service coverage is also limited: only '{only_internet}' is represented, with a churn rate of {only_rate:.2f}%. Therefore this local file cannot support a reliable comparison across internet-service categories."
        )

    else:

        highest_internet = internet_rates.index[0]
        highest_internet_rate = (
            internet_rates.iloc[0]["mean"]
            * 100
        )

        insights.append(
            f"2. Internet-service pattern: '{highest_internet}' has the highest observed churn rate at {highest_internet_rate:.2f}% in this dataset."
        )

    # Insight 3 - payment method
    highest_payment = payment_rates.index[0]
    highest_payment_rate = (
        payment_rates.iloc[0]["mean"]
        * 100
    )

    lowest_payment = payment_rates.index[-1]
    lowest_payment_rate = (
        payment_rates.iloc[-1]["mean"]
        * 100
    )

    insights.append(
        f"3. Payment-method pattern: '{highest_payment}' has the highest observed churn rate at {highest_payment_rate:.2f}%, while '{lowest_payment}' has the lowest at {lowest_payment_rate:.2f}%. The difference is descriptive and should be validated before changing payment policy."
    )

    # Insight 4 - senior citizen
    if len(senior_rates) >= 2:

        yes_rate = (
            senior_rates.loc["Yes", "mean"]
            * 100
            if "Yes" in senior_rates.index
            else np.nan
        )

        no_rate = (
            senior_rates.loc["No", "mean"]
            * 100
            if "No" in senior_rates.index
            else np.nan
        )

        if not np.isnan(yes_rate) and not np.isnan(no_rate):

            insights.append(
                f"4. Senior-citizen pattern: the observed churn rate is {yes_rate:.2f}% for customers marked 'Yes' and {no_rate:.2f}% for customers marked 'No'. This is a descriptive difference, not a causal conclusion."
            )

        else:

            insights.append(
                "4. Senior-citizen coverage is present but does not contain both expected groups, so the comparison is limited."
            )

    else:

        insights.append(
            "4. Senior-citizen coverage contains only one observed category in this dataset, so a group comparison is not reliable."
        )

    # Insight 5 - tenure
    highest_tenure_group = tenure_rates.index[0]
    highest_tenure_rate = (
        tenure_rates.iloc[0]["mean"]
        * 100
    )

    lowest_tenure_group = tenure_rates.index[-1]
    lowest_tenure_rate = (
        tenure_rates.iloc[-1]["mean"]
        * 100
    )

    insights.append(
        f"5. Tenure pattern: the '{highest_tenure_group}' group has the highest observed churn rate at {highest_tenure_rate:.2f}%, while '{lowest_tenure_group}' has the lowest at {lowest_tenure_rate:.2f}%. This suggests tenure is an important predictive signal in this dataset, but it does not establish causation."
    )

    # Write text file.
    with open(
        OUTPUT_FOLDER / "eda_insights.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "CUSTOMER CHURN - FIVE EDA INSIGHTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Overall churn rate: {churn_rate:.2f}%\n\n"
        )

        for insight in insights:

            file.write(
                insight
                + "\n\n"
            )

    return insights, churn_rate



def perform_eda(df):
    """Perform EDA and create the Day 7 outputs."""

    print_section("EXPLORATORY DATA ANALYSIS")

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "\nChurn distribution:"
    )

    print(
        df["Churn Label"].value_counts()
    )

    churn_rate = (
        df["Churn"].mean()
        * 100
    )

    print(
        f"\nOverall churn rate: {churn_rate:.2f}%"
    )

    (
        contract_rates,
        internet_rates,
        payment_rates,
        senior_rates,
        tenure_rates,
    ) = calculate_eda_statistics(
        df
    )

    print(
        "\nChurn rate by contract:"
    )
    print(
        (contract_rates["mean"] * 100).to_string()
    )

    print(
        "\nChurn rate by internet service:"
    )
    print(
        (internet_rates["mean"] * 100).to_string()
    )

    print(
        "\nChurn rate by payment method:"
    )
    print(
        (payment_rates["mean"] * 100).to_string()
    )

    print(
        "\nChurn rate by senior citizen:"
    )
    print(
        (senior_rates["mean"] * 100).to_string()
    )

    print(
        "\nChurn rate by tenure group:"
    )
    print(
        (tenure_rates["mean"] * 100).to_string()
    )

    insights, churn_rate = write_eda_insights(
        df,
        contract_rates,
        internet_rates,
        payment_rates,
        senior_rates,
        tenure_rates,
    )

    # --------------------------------------------------------
    # Chart 1 - churn distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    df["Churn Label"].value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Customer Churn Distribution"
    )

    plt.xlabel(
        "Churn"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "churn_distribution.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 2 - churn by contract
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    (contract_rates["mean"] * 100).sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Churn Rate by Contract"
    )

    plt.xlabel(
        "Churn Rate (%)"
    )

    plt.ylabel(
        "Contract"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "churn_by_contract.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 3 - monthly charges
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    df.boxplot(
        column="Monthly Charges",
        by="Churn Label",
    )

    plt.title(
        "Monthly Charges by Churn"
    )

    plt.suptitle("")

    plt.xlabel(
        "Churn"
    )

    plt.ylabel(
        "Monthly Charges"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "monthly_charges_by_churn.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 4 - payment method
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    (payment_rates["mean"] * 100).sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Churn Rate by Payment Method"
    )

    plt.xlabel(
        "Churn Rate (%)"
    )

    plt.ylabel(
        "Payment Method"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "churn_by_payment_method.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 5 - tenure group
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    tenure_plot = (
        tenure_rates["mean"] * 100
    ).sort_index()

    tenure_plot.plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Tenure Group"
    )

    plt.xlabel(
        "Tenure Group"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "churn_by_tenure_group.png",
        dpi=150,
    )

    plt.close()

    # Save EDA tables.
    contract_rates.to_csv(
        OUTPUT_FOLDER
        / "eda_contract_rates.csv"
    )

    internet_rates.to_csv(
        OUTPUT_FOLDER
        / "eda_internet_rates.csv"
    )

    payment_rates.to_csv(
        OUTPUT_FOLDER
        / "eda_payment_rates.csv"
    )

    senior_rates.to_csv(
        OUTPUT_FOLDER
        / "eda_senior_rates.csv"
    )

    tenure_rates.to_csv(
        OUTPUT_FOLDER
        / "eda_tenure_rates.csv"
    )

    return (
        contract_rates,
        internet_rates,
        payment_rates,
        senior_rates,
        tenure_rates,
        insights,
        churn_rate,
    )


# ============================================================
# 11. SELECT FINAL MODEL FEATURES
# ============================================================


def select_model_features(df):
    """
    Remove leakage and constant predictors.

    Constant predictors are not useful because they contain no
    variation in the supplied training data.
    """

    categorical_features = []
    numerical_features = []
    constant_features = []

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if column not in df.columns:
            continue

        unique_count = (
            df[column]
            .nunique(dropna=True)
        )

        if unique_count <= 1:

            constant_features.append(
                column
            )

        else:

            categorical_features.append(
                column
            )

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    for column in NUMERICAL_FEATURES:

        if column not in df.columns:
            continue

        unique_count = (
            df[column]
            .nunique(dropna=True)
        )

        if unique_count <= 1:

            constant_features.append(
                column
            )

        else:

            numerical_features.append(
                column
            )

    model_features = (
        categorical_features
        + numerical_features
    )

    X = df[
        model_features
    ].copy()

    y = df[
        "Churn"
    ].copy()

    return (
        X,
        y,
        categorical_features,
        numerical_features,
        constant_features,
    )


# ============================================================
# 12. PREPROCESSOR
# ============================================================


def create_preprocessor(
    categorical_features,
    numerical_features,
):
    """Create the reusable preprocessing pipeline."""

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
        ]
    )


# ============================================================
# 13. TRAIN AND EVALUATE ONE MODEL
# ============================================================


def train_and_evaluate(
    model_name,
    estimator,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train one model inside a full preprocessing pipeline."""

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    print(
        f"\nTraining {model_name}..."
    )

    model_pipeline.fit(
        X_train,
        y_train,
    )

    predictions = model_pipeline.predict(
        X_test
    )

    probabilities = (
        model_pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return (
        model_pipeline,
        predictions,
        probabilities,
        metrics,
    )


# ============================================================
# 14. MODEL TRAINING
# ============================================================


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_features,
    numerical_features,
):
    """Train Logistic Regression and Random Forest."""

    print_section("MODEL TRAINING")

    # Create a separate preprocessor for each pipeline so that
    # the models do not accidentally share fitted state.
    logistic_preprocessor = create_preprocessor(
        categorical_features,
        numerical_features,
    )

    random_forest_preprocessor = create_preprocessor(
        categorical_features,
        numerical_features,
    )

    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    (
        logistic_pipeline,
        logistic_predictions,
        logistic_probabilities,
        logistic_metrics,
    ) = train_and_evaluate(
        "Logistic Regression",
        logistic_model,
        logistic_preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    (
        forest_pipeline,
        forest_predictions,
        forest_probabilities,
        forest_metrics,
    ) = train_and_evaluate(
        "Random Forest",
        random_forest_model,
        random_forest_preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison = pd.DataFrame(
        [
            logistic_metrics,
            forest_metrics,
        ]
    )

    comparison = (
        comparison
        .sort_values(
            by=[
                "F1 Score",
                "Recall",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print_section("MODEL COMPARISON")

    print(
        comparison.to_string(
            index=False
        )
    )

    trained_models = {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": forest_pipeline,
    }

    predictions = {
        "Logistic Regression": logistic_predictions,
        "Random Forest": forest_predictions,
    }

    probabilities = {
        "Logistic Regression": logistic_probabilities,
        "Random Forest": forest_probabilities,
    }

    return (
        trained_models,
        predictions,
        probabilities,
        comparison,
    )


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================


def save_test_predictions(
    selected_model_name,
    selected_model,
    X_test,
    y_test,
):
    """Save held-out test predictions and risk categories."""

    probabilities = (
        selected_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        selected_model
        .predict(
            X_test
        )
    )

    output = X_test.copy()

    output["Actual_Churn"] = (
        y_test.values
    )

    output["Predicted_Churn"] = (
        predictions
    )

    output["Churn_Probability"] = (
        probabilities
    )

    output["Risk_Category"] = pd.cut(
        probabilities,
        bins=[
            -np.inf,
            0.30,
            0.60,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        right=False,
    )

    output.to_csv(
        OUTPUT_FOLDER
        / "test_predictions.csv",
        index=False,
    )

    print(
        f"Test predictions saved for {selected_model_name}."
    )


# ============================================================
# 16. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================


def save_random_forest_importance(
    random_forest_pipeline,
):
    """Save Random Forest feature importance as a table and chart."""

    preprocessor = (
        random_forest_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    forest = (
        random_forest_pipeline
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance_values = (
        forest
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        OUTPUT_FOLDER
        / "random_forest_feature_importance.csv",
        index=False,
    )

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.title(
        "Top Random Forest Feature Importances"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "random_forest_feature_importance.png",
        dpi=150,
    )

    plt.close()

    return importance_df


# ============================================================
# 17. WRITE DAY 6 PROBLEM STATEMENT
# ============================================================


def write_problem_statement(
    df,
    churn_rate,
):
    """Create the Day 6 one-page business problem statement."""

    path = (
        OUTPUT_FOLDER
        / "day6_problem_statement.md"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Day 6 - Customer Churn Problem Statement\n\n"
        )

        file.write(
            "## Business Problem\n\n"
        )

        file.write(
            "The goal of this project is to predict which customers are likely to leave the service. The prediction can help a business prioritize customers for retention review before deciding what action, if any, is appropriate.\n\n"
        )

        file.write(
            "## Potential Business Value\n\n"
        )

        file.write(
            "A churn prediction system can help the business focus limited retention resources on customers who show higher estimated churn risk. This can support earlier customer-service review, targeted communication, service education, or other retention programs. The model should support human review rather than automatically decide customer treatment.\n\n"
        )

        file.write(
            "## Possible Actions\n\n"
        )

        file.write(
            "- **High risk:** prioritize a retention review and consider proactive outreach.\n"
        )

        file.write(
            "- **Medium risk:** consider proactive engagement, service/support follow-up, or targeted communication.\n"
        )

        file.write(
            "- **Low risk:** continue normal service and monitoring.\n\n"
        )

        file.write(
            "These are example actions for the academic project and should not be treated as approved business policy.\n\n"
        )

        file.write(
            "## Dataset Context\n\n"
        )

        file.write(
            f"The local project file contains {len(df):,} customer records after cleaning, with an observed churn rate of {churn_rate:.2f}%.\n\n"
        )

        file.write(
            "## Target\n\n"
        )

        file.write(
            "`Churn Label` is converted into a binary target called `Churn`, where Yes = 1 and No = 0.\n\n"
        )

        file.write(
            "## Success Measures\n\n"
        )

        file.write(
            "The project compares Logistic Regression and Random Forest using Accuracy, Precision, Recall, F1 Score, and ROC-AUC. F1 Score is used as the primary model-selection metric with Recall as the tie-breaker because the project needs to balance missed churners against false alarms.\n"
        )


# ============================================================
# 18. WRITE DAY 10 EVALUATION SUMMARY
# ============================================================


def write_evaluation_summary(
    df,
    comparison,
    selected_model_name,
    selected_predictions,
    y_test,
    selected_probabilities,
):
    """Create the Day 10 business evaluation document."""

    selected_row = comparison[
        comparison["Model"] == selected_model_name
    ].iloc[0]

    cm = confusion_matrix(
        y_test,
        selected_predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    baseline_accuracy = max(
        df["Churn"].mean(),
        1 - df["Churn"].mean(),
    )

    path = (
        OUTPUT_FOLDER
        / "day10_evaluation_summary.md"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Day 10 - Model Evaluation and Business Value\n\n"
        )

        file.write(
            "## Model Comparison\n\n"
        )

        file.write(
            "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n"
        )

        file.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for _, row in comparison.iterrows():

            file.write(
                "| "
                f"{row['Model']} | "
                f"{row['Accuracy']:.4f} | "
                f"{row['Precision']:.4f} | "
                f"{row['Recall']:.4f} | "
                f"{row['F1 Score']:.4f} | "
                f"{row['ROC-AUC']:.4f} |\n"
            )

        file.write(
            "\n## Selected Model\n\n"
        )

        file.write(
            f"**{selected_model_name}** was selected because it has the highest F1 Score in this run, with Recall used as the tie-breaker.\n\n"
        )

        file.write(
            "## Why Accuracy Alone Is Not Enough\n\n"
        )

        file.write(
            "Accuracy measures the percentage of all test customers classified correctly. In a churn project, however, the business also needs to understand how many actual churners were found and how many customers were flagged incorrectly. Precision measures the proportion of predicted churners who actually churned; Recall measures the proportion of actual churners that the model detected; F1 combines Precision and Recall. Therefore, looking only at Accuracy could hide an important business trade-off.\n\n"
        )

        file.write(
            f"For context, the majority-class baseline accuracy in this dataset is approximately {baseline_accuracy:.4f}. The selected model's test Accuracy is {selected_row['Accuracy']:.4f}.\n\n"
        )

        file.write(
            "## Selected Model Test Confusion Matrix\n\n"
        )

        file.write(
            f"- True Negatives: {tn}\n"
        )

        file.write(
            f"- False Positives: {fp}\n"
        )

        file.write(
            f"- False Negatives: {fn}\n"
        )

        file.write(
            f"- True Positives: {tp}\n\n"
        )

        file.write(
            "False negatives represent customers who churned but were classified as non-churners, which can mean missed retention opportunities. False positives represent customers who were flagged as churn risks but did not churn, which can consume retention resources unnecessarily. The appropriate balance depends on business costs and capacity.\n\n"
        )

        file.write(
            "## Example Business Action\n\n"
        )

        file.write(
            "Use the model to prioritize review rather than to automatically make customer decisions. High-risk cases can be reviewed first for appropriate outreach, service issues, or other approved retention options. Medium-risk cases can receive lower-intensity follow-up, while low-risk cases can remain in normal monitoring.\n\n"
        )

        file.write(
            "## Risk Categories Used by the Demo App\n\n"
        )

        file.write(
            "- Low: churn probability < 30%\n"
        )

        file.write(
            "- Medium: 30% to less than 60%\n"
        )

        file.write(
            "- High: churn probability >= 60%\n\n"
        )

        file.write(
            "These thresholds are initial demonstration thresholds, not validated business thresholds.\n\n"
        )

        file.write(
            "## Limitations\n\n"
        )

        file.write(
            "The local dataset is not the complete original public Telco Customer Churn table. The current project file contains a restricted subset, including limited category coverage for some variables. Model results therefore describe this local dataset and should not be generalized automatically to a different population.\n"
        )


# ============================================================
# 19. GENERATE A VALID JUPYTER NOTEBOOK
# ============================================================


def make_notebook(cells):
    """Build a simple nbformat v4 notebook without extra dependencies."""

    notebook_cells = []

    for cell_type, source in cells:

        notebook_cells.append(
            {
                "cell_type": cell_type,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(True),
            }
        )

    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }



def create_eda_notebook(insights):
    """Create the Day 7 EDA notebook deliverable."""

    cells = [

        (
            "markdown",
            "# Day 7 - Exploratory Data Analysis\n\n"
            "This notebook documents customer demographics, behavior, churn patterns, and five project insights."
        ),

        (
            "code",
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "\n"
            "project_root = Path.cwd()\n"
            "data_path = project_root / 'data' / 'churn IBM dataset.csv'\n"
            "df = pd.read_csv(data_path, sep='\\t')\n"
            "df.head()"
        ),

        (
            "code",
            "print('Shape:', df.shape)\n"
            "print('Columns:', df.columns.tolist())\n"
            "print('Missing values:')\n"
            "print(df.isnull().sum().sort_values(ascending=False).head(15))"
        ),

        (
            "code",
            "df['Churn'] = df['Churn Label'].map({'No': 0, 'Yes': 1})\n"
            "print('Churn distribution:')\n"
            "print(df['Churn Label'].value_counts())\n"
            "print('Churn rate:', round(df['Churn'].mean() * 100, 2), '%')"
        ),

        (
            "code",
            "for column in ['Contract', 'Internet Service', 'Payment Method', 'Senior Citizen']:\n"
            "    print('\\n', column)\n"
            "    print((df.groupby(column)['Churn'].mean() * 100).sort_values(ascending=False))"
        ),

        (
            "markdown",
            "## Five documented insights from the training run\n\n"
            + "\n".join(
                f"- {insight}"
                for insight in insights
            )
        ),

        (
            "markdown",
            "## Interpretation reminder\n\n"
            "EDA shows associations in the available data. These patterns should not automatically be interpreted as causal relationships."
        ),
    ]

    notebook = make_notebook(
        cells
    )

    with open(
        OUTPUT_FOLDER
        / "day7_eda_notebook.ipynb",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            notebook,
            file,
            indent=2,
        )



def create_feature_engineering_notebook(
    categorical_features,
    numerical_features,
    constant_features,
):
    """Create the Day 8 feature-engineering notebook deliverable."""

    cells = [

        (
            "markdown",
            "# Day 8 - Feature Engineering\n\n"
            "This notebook explains the engineered variables, categorical encoding, numerical scaling, and leakage prevention used by the project."
        ),

        (
            "markdown",
            "## Engineered features\n\n"
            "1. **AvgMonthlySpend** = Total Charges / Tenure Months when tenure is greater than zero.\n"
            "2. **TenureGroup** = grouped customer tenure.\n"
            "3. **IsNewCustomer** = Yes for tenure of six months or less.\n"
            "4. **TotalServices** = number of service fields with a Yes value."
        ),

        (
            "code",
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')\n"
            "df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce')\n"
            "df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')\n"
            "df['AvgMonthlySpend'] = np.where(\n"
            "    df['Tenure Months'] > 0,\n"
            "    df['Total Charges'] / df['Tenure Months'],\n"
            "    df['Monthly Charges']\n"
            ")\n"
            "df['TenureGroup'] = pd.cut(\n"
            "    df['Tenure Months'],\n"
            "    bins=[-1, 6, 12, 24, 48, np.inf],\n"
            "    labels=['0-6 months', '7-12 months', '13-24 months', '25-48 months', '49+ months']\n"
            ")\n"
            "df['IsNewCustomer'] = np.where(df['Tenure Months'] <= 6, 'Yes', 'No')\n"
            "df[['Tenure Months', 'Total Charges', 'AvgMonthlySpend', 'TenureGroup', 'IsNewCustomer']].head()"
        ),

        (
            "markdown",
            "## Encoding and scaling\n\n"
            "Categorical variables are imputed with the most frequent category and one-hot encoded. Numerical variables are median-imputed and standardized. These transformations are fit inside the model pipelines so that test data does not influence training preprocessing."
        ),

        (
            "markdown",
            "## Leakage prevention\n\n"
            "Excluded fields: `CustomerID`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, and `Churn Reason`. These fields are not used as predictive inputs."
        ),

        (
            "markdown",
            "## Final model feature lists\n\n"
            "### Categorical\n"
            + "\n".join(
                f"- {feature}"
                for feature in categorical_features
            )
            + "\n\n### Numerical\n"
            + "\n".join(
                f"- {feature}"
                for feature in numerical_features
            )
        ),

        (
            "markdown",
            "## Constant columns in this local dataset\n\n"
            + (
                "\n".join(
                    f"- {feature}"
                    for feature in constant_features
                )
                if constant_features
                else "No constant model candidate columns were removed."
            )
        ),
    ]

    notebook = make_notebook(
        cells
    )

    with open(
        OUTPUT_FOLDER
        / "day8_feature_engineering_notebook.ipynb",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            notebook,
            file,
            indent=2,
        )


# ============================================================
# 20. CREATE README
# ============================================================


def write_readme(
    comparison,
    selected_model_name,
    feature_count,
    churn_rate,
    constant_features,
):
    """Create a GitHub-ready README."""

    readme_path = (
        PROJECT_ROOT
        / "README.md"
    )

    with open(
        readme_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Customer Churn Prediction - Week 2\n\n"
        )

        file.write(
            "An end-to-end machine-learning project for predicting customer churn and presenting the prediction through a Streamlit interface.\n\n"
        )

        file.write(
            "## Business Problem\n\n"
        )

        file.write(
            "Which customers are likely to leave, and what action could the business take? The project uses churn probability to support prioritization of retention review.\n\n"
        )

        file.write(
            "## Dataset\n\n"
        )

        file.write(
            "Local dataset filename: `data/churn IBM dataset.csv`.\n\n"
        )

        file.write(
            "The public Telco Customer Churn dataset is associated with IBM sample data and is also distributed through public mirrors such as Kaggle. The local file used in this project is a restricted subset/export, so its results should be interpreted as results for this local data rather than automatically generalized to the complete public dataset.\n\n"
        )

        file.write(
            "Public references:\n"
            "- IBM Telco customer churn sample: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113\n"
            "- Kaggle Telco Customer Churn mirror: https://www.kaggle.com/blastchar/telco-customer-churn\n\n"
        )

        file.write(
            "## Project Workflow\n\n"
        )

        file.write(
            "1. Business problem definition\n"
            "2. Exploratory data analysis\n"
            "3. Feature engineering\n"
            "4. Leakage prevention\n"
            "5. Categorical encoding and numerical scaling\n"
            "6. Logistic Regression\n"
            "7. Random Forest\n"
            "8. Model comparison\n"
            "9. Business evaluation\n"
            "10. Streamlit prediction interface\n\n"
        )

        file.write(
            "## Models\n\n"
        )

        file.write(
            "- Logistic Regression\n"
            "- Random Forest\n\n"
        )

        file.write(
            "XGBoost was treated as optional, so the project does not add it as a mandatory dependency.\n\n"
        )

        file.write(
            "## Model Comparison\n\n"
        )

        file.write(
            "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n"
        )

        file.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for _, row in comparison.iterrows():

            file.write(
                "| "
                f"{row['Model']} | "
                f"{row['Accuracy']:.4f} | "
                f"{row['Precision']:.4f} | "
                f"{row['Recall']:.4f} | "
                f"{row['F1 Score']:.4f} | "
                f"{row['ROC-AUC']:.4f} |\n"
            )

        file.write(
            f"\nSelected model: **{selected_model_name}**\n\n"
        )

        file.write(
            f"Observed churn rate in the local dataset: **{churn_rate:.2f}%**.\n\n"
        )

        file.write(
            f"Final model feature count: **{feature_count}** before one-hot expansion.\n\n"
        )

        if constant_features:

            file.write(
                "Constant candidate features removed from modeling in this local dataset: "
                + ", ".join(constant_features)
                + ".\n\n"
            )

        file.write(
            "## Why Accuracy Is Not Enough\n\n"
        )

        file.write(
            "Churn prediction involves two types of mistakes: missing customers who actually churn and incorrectly flagging customers who stay. Precision and Recall expose those errors more clearly than Accuracy alone, while F1 balances Precision and Recall.\n\n"
        )

        file.write(
            "## Business Actions\n\n"
        )

        file.write(
            "- High risk: prioritize retention review and proactive outreach.\n"
            "- Medium risk: consider targeted engagement or support follow-up.\n"
            "- Low risk: continue normal service and monitoring.\n\n"
        )

        file.write(
            "These are example project actions, not approved business policy.\n\n"
        )

        file.write(
            "## Streamlit\n\n"
        )

        file.write(
            "The same Python file contains the Streamlit interface. First train the model, then launch the app.\n\n"
        )

        file.write(
            "```bash\n"
            "python \"customer churn.py\"\n"
            "streamlit run \"customer churn.py\"\n"
            "```\n\n"
        )

        file.write(
            "## Generated Deliverables\n\n"
        )

        file.write(
            "- `outputs/day6_problem_statement.md`\n"
            "- `outputs/day7_eda_notebook.ipynb`\n"
            "- `outputs/day8_feature_engineering_notebook.ipynb`\n"
            "- `outputs/day10_evaluation_summary.md`\n"
            "- `outputs/model_comparison.csv`\n"
            "- `outputs/test_predictions.csv`\n"
            "- `outputs/best_churn_model.joblib`\n"
            "- `outputs/model_metadata.json`\n"
            "- EDA and feature-importance charts\n\n"
        )

        file.write(
            "## Important Data Limitation\n\n"
        )

        file.write(
            "The local project dataset contains only a subset of the categories present in the complete public Telco Churn dataset. In particular, some fields have limited category coverage. The generated evaluation report documents this limitation.\n\n"
        )

        file.write(
            "## GitHub\n\n"
        )

        file.write(
            "Before publishing, verify the dataset license/usage terms and do not commit secrets or personal credentials.\n\n"
        )

        file.write(
            "Suggested commands:\n\n"
        )

        file.write(
            "```bash\n"
            "git init\n"
            "git add .\n"
            "git commit -m \"Add customer churn prediction project\"\n"
            "git branch -M main\n"
            "git remote add origin <YOUR_GITHUB_REPOSITORY_URL>\n"
            "git push -u origin main\n"
            "```\n"
        )


# ============================================================
# 21. CREATE REQUIREMENTS AND GITIGNORE
# ============================================================


def write_project_files():
    """Create small project files needed for GitHub."""

    requirements = """pandas>=2.0\nnumpy>=1.24\nmatplotlib>=3.7\nscikit-learn>=1.3\njoblib>=1.3\nstreamlit>=1.30\n"""

    with open(
        PROJECT_ROOT / "requirements.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            requirements
        )

    gitignore = """# Python\n__pycache__/\n*.py[cod]\n*.pyo\n\n# Virtual environments\n.venv/\nvenv/\nenv/\n\n# VS Code\n.vscode/\n\n# Streamlit secrets\n.streamlit/secrets.toml\n\n# OS files\n.DS_Store\nThumbs.db\n\n# Jupyter checkpoints\n.ipynb_checkpoints/\n"""

    with open(
        PROJECT_ROOT / ".gitignore",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            gitignore
        )


# ============================================================
# 22. SAVE MODEL METADATA
# ============================================================


def save_metadata(
    selected_model_name,
    categorical_features,
    numerical_features,
    constant_features,
    df,
):
    """Save information needed by the Streamlit app."""

    feature_options = {}

    # Save options for all user-facing fields, including fields
    # that may have been removed from the final model because they
    # were constant in the supplied training data.
    all_user_categorical = list(dict.fromkeys(
        CATEGORICAL_FEATURES
    ))

    all_user_numerical = list(dict.fromkeys(
        NUMERICAL_FEATURES
    ))

    for column in (
        all_user_categorical
        + all_user_numerical
    ):

        if column in df.columns:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                feature_options[column] = {
                    "type": "numeric",
                    "min": float(
                        df[column].min()
                    ),
                    "max": float(
                        df[column].max()
                    ),
                    "median": float(
                        df[column].median()
                    ),
                }

            else:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                feature_options[column] = {
                    "type": "categorical",
                    "values": values,
                }

    metadata = {
        "selected_model": selected_model_name,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "constant_features_removed": constant_features,
        "feature_options": feature_options,
        "risk_thresholds": {
            "low_max": 0.30,
            "medium_max": 0.60,
        },
        "risk_actions": {
            "High": (
                "Prioritize the customer for retention review "
                "and proactive outreach."
            ),
            "Medium": (
                "Consider proactive engagement, support follow-up, "
                "or targeted communication."
            ),
            "Low": (
                "Continue normal service and monitoring."
            ),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    return metadata


# ============================================================
# 23. TRAINING MODE
# ============================================================


def run_training():
    """Run the full Week 2 training and documentation workflow."""

    print_section(
        "WEEK 2 - CUSTOMER CHURN PREDICTION"
    )

    # --------------------------------------------------------
    # Load and validate
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal dataset shape: {df.shape}"
    )

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_data(
        df
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = create_features(
        df
    )

    # --------------------------------------------------------
    # Save cleaned data and quality report
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FOLDER
        / "cleaned_churn_data.csv",
        index=False,
    )

    save_data_quality_report(
        df
    )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    (
        contract_rates,
        internet_rates,
        payment_rates,
        senior_rates,
        tenure_rates,
        insights,
        churn_rate,
    ) = perform_eda(
        df
    )

    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    (
        X,
        y,
        categorical_features,
        numerical_features,
        constant_features,
    ) = select_model_features(
        df
    )

    print_section(
        "FEATURE SELECTION AND LEAKAGE CHECK"
    )

    print(
        "Excluded leakage/outcome columns:"
    )

    for column in LEAKAGE_COLUMNS:

        print(
            f" - {column}"
        )

    if constant_features:

        print(
            "\nConstant candidate features removed:"
        )

        for column in constant_features:

            print(
                f" - {column}"
            )

    print(
        "\nFinal model features:"
    )

    for column in X.columns:

        print(
            f" - {column}"
        )

    save_feature_engineering_notes(
        df,
        categorical_features,
        numerical_features,
        constant_features,
    )

    # --------------------------------------------------------
    # Save Day 6 problem statement
    # --------------------------------------------------------

    write_problem_statement(
        df,
        churn_rate,
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Train both models
    # --------------------------------------------------------

    (
        trained_models,
        prediction_map,
        probability_map,
        comparison,
    ) = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features,
        numerical_features,
    )

    comparison.to_csv(
        OUTPUT_FOLDER
        / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    selected_model_name = comparison.loc[
        0,
        "Model",
    ]

    selected_model = trained_models[
        selected_model_name
    ]

    selected_predictions = prediction_map[
        selected_model_name
    ]

    selected_probabilities = probability_map[
        selected_model_name
    ]

    print_section(
        "MODEL SELECTION"
    )

    print(
        f"Selected model: {selected_model_name}"
    )

    print(
        "Selection rule: highest F1 Score, with Recall as tie-breaker."
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        selected_model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_test_predictions(
        selected_model_name,
        selected_model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Random Forest importance
    # --------------------------------------------------------

    importance_df = save_random_forest_importance(
        trained_models["Random Forest"]
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        selected_model_name,
        categorical_features,
        numerical_features,
        constant_features,
        df,
    )

    # --------------------------------------------------------
    # Save Day 7 and Day 8 notebooks
    # --------------------------------------------------------

    create_eda_notebook(
        insights
    )

    create_feature_engineering_notebook(
        categorical_features,
        numerical_features,
        constant_features,
    )

    # --------------------------------------------------------
    # Save Day 10 evaluation summary
    # --------------------------------------------------------

    write_evaluation_summary(
        df,
        comparison,
        selected_model_name,
        selected_predictions,
        y_test,
        selected_probabilities,
    )

    # --------------------------------------------------------
    # Project-level files
    # --------------------------------------------------------

    write_project_files()

    write_readme(
        comparison,
        selected_model_name,
        len(X.columns),
        churn_rate,
        constant_features,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_section(
        "PROJECT COMPLETED SUCCESSFULLY"
    )

    print(
        f"Selected model: {selected_model_name}"
    )

    print(
        f"Observed churn rate: {churn_rate:.2f}%"
    )

    print(
        "\nModel comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nGenerated project files:"
    )

    generated_files = sorted(
        OUTPUT_FOLDER.iterdir()
    )

    for file in generated_files:

        print(
            f" - {file.name}"
        )

    print(
        "\nREADME.md and requirements.txt were created in the project root."
    )

    print(
        "\nTo start the final application, run:"
    )

    print(
        'streamlit run "customer churn.py"'
    )


# ============================================================
# 24. STREAMLIT MODE
# ============================================================


def get_streamlit_context():
    """Return the current Streamlit script context when available."""

    try:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx
        )

        return get_script_run_ctx()

    except Exception:

        return None



def running_in_streamlit():
    """Detect whether this file is being executed by Streamlit."""

    if get_streamlit_context() is not None:
        return True

    # Some Streamlit versions expose a server environment value.
    streamlit_env_keys = [
        "STREAMLIT_SERVER_PORT",
        "STREAMLIT_RUNTIME",
    ]

    return any(
        key in os.environ
        for key in streamlit_env_keys
    )



def load_streamlit_resources():
    """Load model and metadata for the Streamlit application."""

    import streamlit as st

    if not MODEL_PATH.exists():

        st.error(
            "The trained model was not found."
        )

        st.info(
            "First run the training command in the terminal:\n\n"
            'python "customer churn.py"'
        )

        st.stop()

    if not METADATA_PATH.exists():

        st.error(
            "Model metadata was not found."
        )

        st.info(
            "Run the training script once before launching Streamlit."
        )

        st.stop()

    @st.cache_resource
    def load_model():

        return joblib.load(
            MODEL_PATH
        )

    @st.cache_data
    def load_metadata():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    return (
        load_model(),
        load_metadata(),
    )



def streamlit_app():
    """Build the final interactive churn prediction application."""

    import streamlit as st

    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📉",
        layout="wide",
    )

    model, metadata = load_streamlit_resources()

    selected_model = metadata.get(
        "selected_model",
        "Unknown",
    )

    feature_options = metadata.get(
        "feature_options",
        {},
    )

    categorical_features = metadata.get(
        "categorical_features",
        [],
    )

    numerical_features = metadata.get(
        "numerical_features",
        [],
    )

    risk_actions = metadata.get(
        "risk_actions",
        {},
    )

    low_max = metadata.get(
        "risk_thresholds",
        {},
    ).get(
        "low_max",
        0.30,
    )

    medium_max = metadata.get(
        "risk_thresholds",
        {},
    ).get(
        "medium_max",
        0.60,
    )

    st.title(
        "📉 Customer Churn Prediction"
    )

    st.write(
        "Enter a customer's current information to estimate the probability of churn and assign a Low, Medium, or High risk category."
    )

    st.info(
        "This application is a project demonstration. The prediction should support human review rather than automatically decide customer treatment."
    )

    with st.expander(
        "About this model"
    ):

        st.write(
            f"**Selected model:** {selected_model}"
        )

        st.write(
            "**Model-selection rule:** Highest F1 Score, with Recall as the tie-breaker."
        )

        st.write(
            "**Risk thresholds:** Low < 30%, Medium 30%-<60%, High >= 60%."
        )

        st.write(
            "These thresholds are initial demonstration thresholds and are not validated business thresholds."
        )

    # --------------------------------------------------------
    # Form
    # --------------------------------------------------------

    with st.form(
        "customer_form"
    ):

        st.subheader(
            "Customer Profile"
        )

        column1, column2, column3 = st.columns(3)

        with column1:

            def categorical_input(
                column_name,
                label=None,
            ):
                values = feature_options.get(
                    column_name,
                    {},
                ).get(
                    "values",
                    [],
                )

                if not values:
                    values = ["Unknown"]

                return st.selectbox(
                    label or column_name,
                    values,
                )

            gender = categorical_input(
                "Gender"
            )

            senior_citizen = categorical_input(
                "Senior Citizen"
            )

            partner = categorical_input(
                "Partner"
            )

            dependents = categorical_input(
                "Dependents"
            )

            contract = categorical_input(
                "Contract"
            )

        with column2:

            phone_service = categorical_input(
                "Phone Service"
            )

            multiple_lines = categorical_input(
                "Multiple Lines"
            )

            internet_service = categorical_input(
                "Internet Service"
            )

            online_security = categorical_input(
                "Online Security"
            )

            online_backup = categorical_input(
                "Online Backup"
            )

        with column3:

            device_protection = categorical_input(
                "Device Protection"
            )

            tech_support = categorical_input(
                "Tech Support"
            )

            streaming_tv = categorical_input(
                "Streaming TV"
            )

            streaming_movies = categorical_input(
                "Streaming Movies"
            )

            paperless_billing = categorical_input(
                "Paperless Billing"
            )

        st.subheader(
            "Customer Account Information"
        )

        account_column1, account_column2, account_column3 = st.columns(3)

        with account_column1:

            payment_method = categorical_input(
                "Payment Method"
            )

        with account_column2:

            tenure_min = int(
                feature_options.get(
                    "Tenure Months",
                    {},
                ).get(
                    "min",
                    0,
                )
            )

            tenure_max = int(
                np.ceil(
                    feature_options.get(
                        "Tenure Months",
                        {},
                    ).get(
                        "max",
                        120,
                    )
                )
            )

            tenure_default = int(
                round(
                    feature_options.get(
                        "Tenure Months",
                        {},
                    ).get(
                        "median",
                        12,
                    )
                )
            )

            tenure_default = max(
                tenure_min,
                min(
                    tenure_default,
                    tenure_max,
                ),
            )

            tenure_months = st.number_input(
                "Tenure Months",
                min_value=tenure_min,
                max_value=max(
                    tenure_max,
                    tenure_min + 1,
                ),
                value=tenure_default,
                step=1,
            )

        with account_column3:

            monthly_options = feature_options.get(
                "Monthly Charges",
                {},
            )

            monthly_min = float(
                monthly_options.get(
                    "min",
                    0,
                )
            )

            monthly_max = float(
                monthly_options.get(
                    "max",
                    200,
                )
            )

            monthly_median = float(
                monthly_options.get(
                    "median",
                    70,
                )
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=max(
                    monthly_max,
                    monthly_min + 1,
                ),
                value=max(
                    monthly_median,
                    0.0,
                ),
                step=1.0,
            )

        total_charges_options = feature_options.get(
            "Total Charges",
            {},
        )

        total_default = float(
            total_charges_options.get(
                "median",
                monthly_charges * max(
                    tenure_months,
                    1,
                ),
            )
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=max(
                total_default,
                0.0,
            ),
            step=10.0,
        )

        submitted = st.form_submit_button(
            "🔍 Predict Churn Risk"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if submitted:

        # Recreate exactly the engineered fields used by training.
        if tenure_months > 0:

            avg_monthly_spend = (
                total_charges
                / tenure_months
            )

        else:

            avg_monthly_spend = (
                monthly_charges
            )

        if tenure_months <= 6:

            tenure_group = "0-6 months"

        elif tenure_months <= 12:

            tenure_group = "7-12 months"

        elif tenure_months <= 24:

            tenure_group = "13-24 months"

        elif tenure_months <= 48:

            tenure_group = "25-48 months"

        else:

            tenure_group = "49+ months"

        is_new_customer = (
            "Yes"
            if tenure_months <= 6
            else "No"
        )

        service_values = [
            phone_service,
            multiple_lines,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]

        total_services = sum(
            value.strip().lower() == "yes"
            for value in service_values
        )

        customer_data = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "TenureGroup": tenure_group,
            "IsNewCustomer": is_new_customer,
            "Tenure Months": tenure_months,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
            "AvgMonthlySpend": avg_monthly_spend,
            "TotalServices": total_services,
        }

        # Only pass exactly the final trained model columns.
        prediction_frame = pd.DataFrame(
            [
                {
                    column: customer_data.get(
                        column,
                        np.nan,
                    )
                    for column in (
                        categorical_features
                        + numerical_features
                    )
                }
            ]
        )

        churn_probability = float(
            model
            .predict_proba(
                prediction_frame
            )[0][1]
        )

        binary_prediction = int(
            model
            .predict(
                prediction_frame
            )[0]
        )

        # Risk category.
        if churn_probability < low_max:

            risk_category = "Low"

        elif churn_probability < medium_max:

            risk_category = "Medium"

        else:

            risk_category = "High"

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%",
            )

        with result2:

            st.metric(
                "Risk Category",
                risk_category,
            )

        with result3:

            predicted_status = (
                "Likely to Churn"
                if binary_prediction == 1
                else "Likely to Stay"
            )

            st.metric(
                "Model Classification",
                predicted_status,
            )

        st.write(
            "Probability scale"
        )

        st.progress(
            churn_probability
        )

        # Risk message.
        if risk_category == "High":

            st.warning(
                "This customer is in the High Risk category."
            )

        elif risk_category == "Medium":

            st.info(
                "This customer is in the Medium Risk category."
            )

        else:

            st.success(
                "This customer is in the Low Risk category."
            )

        st.subheader(
            "Suggested Business Action"
        )

        st.write(
            risk_actions.get(
                risk_category,
                "Review the customer profile.",
            )
        )

        # Customer summary.
        st.subheader(
            "Customer Summary"
        )

        summary = pd.DataFrame(
            {
                "Attribute": [
                    "Gender",
                    "Senior Citizen",
                    "Partner",
                    "Dependents",
                    "Contract",
                    "Tenure Months",
                    "Internet Service",
                    "Payment Method",
                    "Monthly Charges",
                    "Total Charges",
                    "Total Services",
                ],
                "Value": [
                    gender,
                    senior_citizen,
                    partner,
                    dependents,
                    contract,
                    tenure_months,
                    internet_service,
                    payment_method,
                    f"${monthly_charges:.2f}",
                    f"${total_charges:.2f}",
                    total_services,
                ],
            }
        )

        st.table(
            summary
        )

        # Explain constant-feature limitation.
        constant_features = metadata.get(
            "constant_features_removed",
            [],
        )

        if constant_features:

            st.warning(
                "Some customer fields had only one observed value in the training dataset and were therefore excluded from model learning: "
                + ", ".join(constant_features)
                + "."
            )

        st.caption(
            "The 30% and 60% thresholds are initial demonstration thresholds. Business teams should validate them using retention costs, intervention capacity, and historical outcomes before operational use."
        )


# ============================================================
# 25. ENTRY POINT
# ============================================================

if running_in_streamlit():

    streamlit_app()

else:

    run_training()
