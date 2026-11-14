from athlete_model import train_and_select_model


if __name__ == "__main__":
    result = train_and_select_model()

    print("\n========================================")
    print("MODEL DEVELOPMENT AND VALIDATION REPORT")
    print("========================================")
    print("Study title:")
    print(
        "Development and Validation of an Artificial Intelligence–Based "
        "Model for Assessing Psychological Factors Influencing Sports "
        "Performance among University Athletes"
    )
    print("\nBest model:", result["best_model_name"])
    print("MAE:", round(result["metrics"]["mae"], 4))
    print("RMSE:", round(result["metrics"]["rmse"], 4))
    print("R²:", round(result["metrics"]["r2"], 4))
    print("Mean CV R²:", round(result["mean_cv_r2"], 4))
    print("Model artifact:", result["model_path"])
    print("Comparison report:", result["comparison_path"])
    print("Graph and table reports:")
    for report_name, report_path in result["report_paths"].items():
        print(f"  {report_name}: {report_path}")
    print("========================================")
