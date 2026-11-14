from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

FEATURES = [
    "Motivation",
    "Anxiety",
    "Confidence",
    "Stress",
    "Resilience",
    "Mental_Toughness",
]
TARGET = "Performance"
DEFAULT_DATA_PATH = Path("data/athelete_data.csv")
DEFAULT_MODEL_DIR = Path("models")
RESEARCH_TITLE = (
    "Development and Validation of an Artificial Intelligence–Based Model "
    "for Assessing Psychological Factors Influencing Sports Performance "
    "among University Athletes"
)


def load_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset and ensure the required columns exist."""
    path = Path(data_path)
    df = pd.read_csv(path)

    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create the feature matrix and target vector."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_model_registry() -> Dict[str, object]:
    """Create the list of candidate regression models for comparison."""
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        ),
    }


def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _model_parameters(name: str, model) -> Dict[str, object]:
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        parameters = estimator.get_params(deep=False)
    else:
        parameters = model.get_params(deep=False)
    return {
        "Model": name,
        "Algorithm": estimator.__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__,
        "Hyperparameters": str(parameters),
    }


def generate_analysis_reports(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_models: Dict[str, object],
    results_df: pd.DataFrame,
    best_model_name: str,
    model_dir: Path,
) -> Dict[str, str]:
    """Create CSV tables and publication-ready plots for the regression analysis."""
    report_paths = {}

    descriptive = df[FEATURES + [TARGET]].describe().T.reset_index().rename(columns={"index": "Variable"})
    descriptive.to_csv(model_dir / "descriptive_statistics.csv", index=False)
    report_paths["descriptive_statistics"] = str(model_dir / "descriptive_statistics.csv")

    split_summary = pd.DataFrame([{
        "Dataset Rows": len(df),
        "Training Rows": len(X_train),
        "Test Rows": len(X_test),
        "Training Percent": round(len(X_train) / len(df) * 100, 2),
        "Test Percent": round(len(X_test) / len(df) * 100, 2),
        "Random State": 42,
    }])
    split_summary.to_csv(model_dir / "train_test_split.csv", index=False)
    report_paths["train_test_split"] = str(model_dir / "train_test_split.csv")

    configurations = [_model_parameters(name, model) for name, model in trained_models.items()]
    pd.DataFrame(configurations).to_csv(model_dir / "model_configurations.csv", index=False)
    report_paths["model_configurations"] = str(model_dir / "model_configurations.csv")

    best_model = trained_models[best_model_name]
    actual = y_test.reset_index(drop=True)
    predicted_by_model = pd.DataFrame({"Actual": actual})
    for name, model in trained_models.items():
        predicted_by_model[name] = model.predict(X_test)

    predicted_by_model["Sample"] = np.arange(len(actual)) + 1
    predicted_by_model["Residual"] = predicted_by_model["Actual"] - predicted_by_model[best_model_name]
    predicted_by_model.to_csv(model_dir / "actual_vs_predicted.csv", index=False)
    report_paths["actual_vs_predicted"] = str(model_dir / "actual_vs_predicted.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_data = results_df.sort_values("Model")
    for ax, metric, title in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["Mean Absolute Error", "Root Mean Squared Error", "R-squared"],
    ):
        ax.barh(plot_data["Model"], plot_data[metric], color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(model_dir / "model_performance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report_paths["model_performance_comparison"] = str(model_dir / "model_performance_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(predicted_by_model["Actual"], predicted_by_model[best_model_name], alpha=0.85, color="#E45756")
    limits = [
        min(predicted_by_model["Actual"].min(), predicted_by_model[best_model_name].min()),
        max(predicted_by_model["Actual"].max(), predicted_by_model[best_model_name].max()),
    ]
    ax.plot(limits, limits, linestyle="--", color="#333333", label="Perfect prediction")
    ax.set_xlabel("Actual Performance")
    ax.set_ylabel("Predicted Performance")
    ax.set_title(f"Actual vs Predicted — {best_model_name}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(model_dir / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report_paths["actual_vs_predicted_graph"] = str(model_dir / "actual_vs_predicted.png")

    correlation = df[FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.columns)), correlation.columns)
    for row in range(len(correlation.columns)):
        for column in range(len(correlation.columns)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Heatmap of Input Variables")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(model_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report_paths["correlation_heatmap"] = str(model_dir / "correlation_heatmap.png")

    try:
        scoring = "neg_mean_absolute_error"
        importance = permutation_importance(best_model, X_test, y_test, n_repeats=min(30, max(5, len(y_test))),
                                            random_state=42, scoring=scoring)
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importance.importances_mean,
        }).sort_values("Importance", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"Feature": [], "Importance": []})
    if not importance_df.empty:
        importance_df.to_csv(model_dir / "feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#72B7B2")
        ax.invert_yaxis()
        ax.set_xlabel("Permutation importance (change in MAE)")
        ax.set_title(f"Feature Importance — {best_model_name}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(model_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        report_paths["feature_importance"] = str(model_dir / "feature_importance.csv")
        report_paths["feature_importance_graph"] = str(model_dir / "feature_importance.png")

    residuals = predicted_by_model["Residual"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(residuals, bins=max(3, min(10, len(residuals))), color="#F58518", edgecolor="white")
    axes[0].axvline(0, color="#333333", linestyle="--")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Actual - Predicted")
    axes[0].set_ylabel("Frequency")
    axes[1].scatter(predicted_by_model[best_model_name], residuals, color="#54A24B", alpha=0.85)
    axes[1].axhline(0, color="#333333", linestyle="--")
    axes[1].set_title("Residual Error Analysis")
    axes[1].set_xlabel("Predicted Performance")
    axes[1].set_ylabel("Residual")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(model_dir / "residual_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report_paths["residual_analysis"] = str(model_dir / "residual_analysis.png")

    return report_paths


def train_and_select_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    save_reports: bool = True,
) -> Dict[str, object]:
    """Train, validate, and save the best-performing athlete performance model."""
    df = load_dataset(data_path)
    if len(df) < 4:
        raise ValueError("At least 4 rows are required for model training and evaluation.")

    X, y = prepare_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=max(0.20, min(0.30, 2 / len(df))),
        random_state=42,
    )

    model_registry = build_model_registry()
    results = []
    trained_models = {}
    cv_results = {}

    for name, model in model_registry.items():
        model.fit(X_train, y_train)
        metrics = _evaluate_model(model, X_test, y_test)
        trained_models[name] = model

        cv_folds = min(5, max(2, len(df) // 2))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        cv_results[name] = cv_scores

        results.append({
            "Model": name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
            "Mean_CV_R2": float(cv_scores.mean()),
            "CV_STD": float(cv_scores.std()),
        })

    results_df = pd.DataFrame(results).sort_values(by="Mean_CV_R2", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_metrics = _evaluate_model(best_model, X_test, y_test)
    cv_scores = cv_results[best_model_name]

    # For small athlete datasets, a simpler linear model is often more stable and
    # less prone to overfitting than more complex tree-based AI models. However, the
    # project design still supports advanced ML comparison and will prefer a non-linear
    # model when the dataset is large enough to justify it.
    if len(df) >= 25 and best_model_name not in ["Linear Regression"]:
        preferred_name = max(
            model_registry.keys(),
            key=lambda name: cv_results[name].mean() if len(cv_results[name]) > 0 else float("-inf"),
        )
        best_model_name = preferred_name
        best_model = trained_models[best_model_name]
        best_metrics = _evaluate_model(best_model, X_test, y_test)
        cv_scores = cv_results[best_model_name]

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "sports_performance_model.pkl"
    joblib.dump(best_model, model_path)

    report_paths = {}
    if save_reports:
        results_df.to_csv(model_dir / "model_comparison.csv", index=False)
        report_paths = generate_analysis_reports(
            df=df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            trained_models=trained_models,
            results_df=results_df,
            best_model_name=best_model_name,
            model_dir=model_dir,
        )

        if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            importance = getattr(best_model, "feature_importances_", None)
            if importance is not None:
                importance_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Importance": importance,
                }).sort_values(by="Importance", ascending=False)
                importance_df.to_csv(model_dir / "feature_importance.csv", index=False)

    summary = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "metrics": {
            "mae": float(best_metrics["mae"]),
            "rmse": float(best_metrics["rmse"]),
            "r2": float(best_metrics["r2"]),
        },
        "cv_scores": cv_scores,
        "mean_cv_r2": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "model_path": str(model_path),
        "comparison_path": str(model_dir / "model_comparison.csv"),
        "report_paths": report_paths,
        "research_title": RESEARCH_TITLE,
    }

    return summary


if __name__ == "__main__":
    result = train_and_select_model()
    print("\nBest model:", result["best_model_name"])
    print("MAE:", round(result["metrics"]["mae"], 4))
    print("RMSE:", round(result["metrics"]["rmse"], 4))
    print("R²:", round(result["metrics"]["r2"], 4))
    print("Mean CV R²:", round(result["mean_cv_r2"], 4))
