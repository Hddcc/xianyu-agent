from benchmarks.intent_accuracy import Prediction, baseline_accuracy, classification_metrics


def test_classification_metrics_include_macro_scores_and_confusion():
    predictions = [
        Prediction("a", "price", "price", "rule", None, 1),
        Prediction("b", "tech", "default", "model", None, 2),
        Prediction("c", "default", "default", "model", None, 3),
        Prediction("d", "no_reply", "no_reply", "model", None, 4),
    ]

    metrics = classification_metrics(predictions)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion"]["tech"]["default"] == 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_baselines_do_not_reuse_model_predictions_for_model_routed_cases():
    cases = [("a", "price"), ("b", "default")]
    predictions = [
        Prediction("a", "price", "price", "rule", None, 1),
        Prediction("b", "default", "tech", "model", None, 2),
    ]

    baselines = baseline_accuracy(cases, predictions)

    assert baselines == {"always_default": 0.5, "rule_only": 1.0}
