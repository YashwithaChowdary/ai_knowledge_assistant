
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import joblib


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "exams.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("STUDENT PERFORMANCE PREDICTION PROJECT")
print("=" * 80)


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\n[1] Checking dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"""
Dataset not found.

Expected location:
{DATA_PATH}

Make sure your file is named:
exams.csv

and is inside:
data/
"""
    )

print(f"Dataset found: {DATA_PATH}")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n[2] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. CHECK EXPECTED COLUMNS
# ============================================================

print("\n[3] Validating columns...")

expected_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score",
]

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(f"- {column}" for column in missing_columns)
    )

print("All expected columns are available.")


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n[4] Checking data types...")

score_columns = [
    "math score",
    "reading score",
    "writing score",
]

for column in score_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print(df.dtypes)


# ============================================================
# 6. DATA QUALITY CHECK
# ============================================================

print("\n[5] Data quality analysis...")

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicate_count}")

print("\nScore ranges:")

for column in score_columns:
    print(
        f"{column}: "
        f"min={df[column].min()}, "
        f"max={df[column].max()}"
    )


# ============================================================
# 7. CLEAN DATA
# ============================================================

print("\n[6] Cleaning data...")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicate_count} duplicate rows.")

# Replace invalid scores with NaN
for column in score_columns:

    invalid_values = (
        (df[column] < 0)
        | (df[column] > 100)
    )

    invalid_count = invalid_values.sum()

    if invalid_count > 0:
        print(
            f"Replacing {invalid_count} invalid values "
            f"in {column} with NaN."
        )

        df.loc[invalid_values, column] = np.nan

# Remove rows where every subject score is missing
df = df.dropna(
    subset=score_columns,
    how="all"
).reset_index(drop=True)

print(f"Rows after cleaning: {len(df)}")


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print("\n[7] Feature engineering...")

# Calculate overall score
df["overall_score"] = df[score_columns].mean(axis=1)

print("\nOverall score statistics:")
print(df["overall_score"].describe().round(2))


# Performance category is for interpretation only.
def performance_category(score):
    if pd.isna(score):
        return "Unknown"

    if score >= 70:
        return "High"

    if score >= 50:
        return "Medium"

    return "Low"


df["performance_category"] = (
    df["overall_score"]
    .apply(performance_category)
)

print("\nPerformance category counts:")
print(df["performance_category"].value_counts())


# ============================================================
# 9. SAVE CLEANED DATA
# ============================================================

cleaned_data_path = (
    OUTPUT_DIR / "cleaned_student_data.csv"
)

df.to_csv(
    cleaned_data_path,
    index=False
)

print(
    f"\nCleaned dataset saved to:\n"
    f"{cleaned_data_path}"
)


# ============================================================
# 10. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[8] Exploratory Data Analysis...")

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe().round(2))

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

print("\nCategorical column distributions:")

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts())


# ============================================================
# 11. EDA OBSERVATIONS
# ============================================================

print("\n" + "=" * 80)
print("EDA ANALYSIS")
print("=" * 80)

overall_mean = df["overall_score"].mean()
overall_median = df["overall_score"].median()
highest_score = df["overall_score"].max()
lowest_score = df["overall_score"].min()

print(f"\nAverage overall score : {overall_mean:.2f}")
print(f"Median overall score  : {overall_median:.2f}")
print(f"Highest overall score : {highest_score:.2f}")
print(f"Lowest overall score  : {lowest_score:.2f}")


# Test preparation analysis
preparation_analysis = (
    df.groupby("test preparation course")
    ["overall_score"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nOverall score by test preparation:")
print(preparation_analysis)


# Gender analysis
gender_analysis = (
    df.groupby("gender")
    ["overall_score"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nOverall score by gender:")
print(gender_analysis)


# Parental education analysis
education_analysis = (
    df.groupby("parental level of education")
    ["overall_score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print("\nOverall score by parental education:")
print(education_analysis)


# Lunch analysis
lunch_analysis = (
    df.groupby("lunch")
    ["overall_score"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nOverall score by lunch category:")
print(lunch_analysis)


# ============================================================
# 12. VISUALIZATIONS
# ============================================================

print("\nCreating visualizations...")


# Overall score distribution
plt.figure(figsize=(9, 6))

plt.hist(
    df["overall_score"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Overall Student Scores")
plt.xlabel("Overall Score")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_score_distribution.png",
    dpi=300
)

plt.show()
plt.close()


# Average subject scores
subject_means = df[
    score_columns
].mean()

plt.figure(figsize=(9, 6))

plt.bar(
    subject_means.index,
    subject_means.values
)

plt.title("Average Score by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "average_subject_scores.png",
    dpi=300
)

plt.show()
plt.close()


# Test preparation
plt.figure(figsize=(9, 6))

preparation_means = (
    df.groupby("test preparation course")
    ["overall_score"]
    .mean()
)

plt.bar(
    preparation_means.index,
    preparation_means.values
)

plt.title("Average Overall Score by Test Preparation")
plt.xlabel("Test Preparation Course")
plt.ylabel("Average Overall Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "score_by_test_preparation.png",
    dpi=300
)

plt.show()
plt.close()


# Gender
plt.figure(figsize=(9, 6))

gender_means = (
    df.groupby("gender")
    ["overall_score"]
    .mean()
)

plt.bar(
    gender_means.index,
    gender_means.values
)

plt.title("Average Overall Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Overall Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "score_by_gender.png",
    dpi=300
)

plt.show()
plt.close()


# Parental education
plt.figure(figsize=(11, 6))

education_means = (
    df.groupby("parental level of education")
    ["overall_score"]
    .mean()
    .sort_values(ascending=False)
)

plt.bar(
    education_means.index,
    education_means.values
)

plt.title(
    "Average Overall Score by Parental Level of Education"
)

plt.xlabel("Parental Level of Education")
plt.ylabel("Average Overall Score")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "score_by_parental_education.png",
    dpi=300
)

plt.show()
plt.close()


# Lunch
plt.figure(figsize=(9, 6))

lunch_means = (
    df.groupby("lunch")
    ["overall_score"]
    .mean()
)

plt.bar(
    lunch_means.index,
    lunch_means.values
)

plt.title("Average Overall Score by Lunch Type")
plt.xlabel("Lunch")
plt.ylabel("Average Overall Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "score_by_lunch.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 13. PREPARE ML DATA
# ============================================================

print("\n[9] Preparing machine learning data...")

# Only use information that could reasonably be considered
# input/student-profile information.
#
# DO NOT use math, reading, or writing score as features because
# they are used to calculate overall_score.

feature_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

target_column = "overall_score"

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nFeatures:")
for feature in feature_columns:
    print(f"- {feature}")

print(f"\nTarget: {target_column}")


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 15. PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            feature_columns
        )
    ]
)


# ============================================================
# 16. BASELINE MODEL — LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression baseline...")

linear_regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

linear_regression_pipeline.fit(
    X_train,
    y_train
)

linear_predictions = (
    linear_regression_pipeline
    .predict(X_test)
)


# ============================================================
# 17. RANDOM FOREST MODEL
# ============================================================

print("Training Random Forest model...")

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_pipeline.fit(
    X_train,
    y_train
)

random_forest_predictions = (
    random_forest_pipeline
    .predict(X_test)
)


# ============================================================
# 18. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):
    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# 19. EVALUATE MODELS
# ============================================================

print("\n[10] Evaluating models...")

linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_predictions
)

random_forest_results = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions
)

results = pd.DataFrame(
    [
        linear_results,
        random_forest_results,
    ]
)

print("\nModel comparison:")
print(
    results.round(4).to_string(index=False)
)

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# 20. SELECT MODEL
# ============================================================

best_index = (
    results["RMSE"]
    .idxmin()
)

best_model_name = results.loc[
    best_index,
    "Model"
]

if best_model_name == "Linear Regression":

    best_model = linear_regression_pipeline
    final_predictions = linear_predictions

else:

    best_model = random_forest_pipeline
    final_predictions = random_forest_predictions


print(
    f"\nSelected model based on lowest RMSE: "
    f"{best_model_name}"
)


# ============================================================
# 21. TEST PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame(
    {
        "Actual Score": y_test.values,
        "Predicted Score": final_predictions,
    }
)

prediction_results["Absolute Error"] = (
    prediction_results["Actual Score"]
    - prediction_results["Predicted Score"]
).abs()

print("\nSample predictions:")
print(
    prediction_results
    .head(10)
    .round(2)
    .to_string(index=False)
)

prediction_results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 22. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.6
)

min_score = min(
    y_test.min(),
    final_predictions.min()
)

max_score = max(
    y_test.max(),
    final_predictions.max()
)

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--"
)

plt.title(
    f"Actual vs Predicted Overall Scores\n"
    f"{best_model_name}"
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 23. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

try:

    trained_preprocessor = (
        random_forest_pipeline
        .named_steps["preprocessor"]
    )

    trained_random_forest = (
        random_forest_pipeline
        .named_steps["model"]
    )

    encoded_feature_names = (
        trained_preprocessor
        .get_feature_names_out()
    )

    feature_importances = (
        trained_random_forest
        .feature_importances_
    )

    feature_importance_df = pd.DataFrame(
        {
            "Feature": encoded_feature_names,
            "Importance": feature_importances,
        }
    ).sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop features:")
    print(
        feature_importance_df
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    feature_importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

except Exception as error:

    print(
        "Feature importance calculation failed:"
    )

    print(error)


# ============================================================
# 24. SAVE TRAINED MODEL
# ============================================================

model_path = (
    OUTPUT_DIR / "student_performance_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print(
    f"\nTrained model saved to:\n"
    f"{model_path}"
)


# ============================================================
# 25. REALISTIC SAMPLE PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE STUDENT PREDICTION")
print("=" * 80)

sample_student = pd.DataFrame(
    [
        {
            "gender": "female",
            "race/ethnicity": "group D",
            "parental level of education": "some college",
            "lunch": "standard",
            "test preparation course": "completed",
        }
    ]
)

sample_prediction = best_model.predict(
    sample_student
)[0]

print("\nStudent information:")

print(
    sample_student.to_string(index=False)
)

print(
    f"\nPredicted overall score: "
    f"{sample_prediction:.2f}"
)

print(
    f"Performance category: "
    f"{performance_category(sample_prediction)}"
)


# ============================================================
# 26. SAVE PROJECT SUMMARY
# ============================================================

summary = f"""
STUDENT PERFORMANCE PREDICTION PROJECT
======================================

Dataset
-------
File: exams.csv
Rows after cleaning: {len(df)}

Target
------
overall_score

The target is calculated as the average of:
- math score
- reading score
- writing score

Input Features
--------------
{chr(10).join('- ' + feature for feature in feature_columns)}

Models
------
1. Linear Regression
2. Random Forest Regressor

Selected Model
--------------
{best_model_name}

Best Model Metrics
------------------
MAE  : {results.loc[best_index, 'MAE']:.4f}
RMSE : {results.loc[best_index, 'RMSE']:.4f}
R2   : {results.loc[best_index, 'R2']:.4f}

Important Limitation
--------------------
The dataset does not contain:
- study hours
- attendance
- previous grades
- longitudinal student history

Therefore, those factors are not used.

Target leakage prevention
-------------------------
Math, reading, and writing scores are not used as
model input features because they directly determine
the overall_score target.

Generated Outputs
-----------------
- cleaned_student_data.csv
- model_comparison.csv
- test_predictions.csv
- feature_importance.csv
- student_performance_model.pkl
- overall_score_distribution.png
- average_subject_scores.png
- score_by_test_preparation.png
- score_by_gender.png
- score_by_parental_education.png
- score_by_lunch.png
- actual_vs_predicted.png
"""

summary_path = OUTPUT_DIR / "project_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(
    f"\nProject summary saved to:\n"
    f"{summary_path}"
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("WEEK 1 STUDENT PERFORMANCE PROJECT COMPLETED")
print("=" * 80)

print("""The complete pipeline has finished.""")