from features.feature_store import load_training_data
from training.baseline_model import build_pipeline


def test_feature_store_has_expected_shape():
    X, y = load_training_data()
    assert len(X) == len(y)
    assert len(X) > 0
    assert "EmployeeNumber" not in X.columns


def test_pipeline_can_be_created():
    X, _ = load_training_data()
    pipeline = build_pipeline(X)
    assert pipeline is not None
