from pathlib import Path

from athlete_model import (
    build_model_registry,
    load_dataset,
    prepare_features_target,
    train_and_select_model,
)


def test_dataset_and_features_are_valid():
    df = load_dataset(Path("data/athelete_data.csv"))
    X, y = prepare_features_target(df)

    assert list(X.columns) == [
        "Motivation",
        "Anxiety",
        "Confidence",
        "Stress",
        "Resilience",
        "Mental_Toughness",
    ]
    assert "Performance" in y.name
    assert len(df) >= 5


def test_training_pipeline_creates_model_artifacts(tmp_path):
    result = train_and_select_model(
        data_path=Path("data/athelete_data.csv"),
        model_dir=tmp_path,
        save_reports=True,
    )

    assert result["best_model_name"] in build_model_registry()
    assert (tmp_path / "sports_performance_model.pkl").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert result["metrics"]["r2"] >= -1.0
