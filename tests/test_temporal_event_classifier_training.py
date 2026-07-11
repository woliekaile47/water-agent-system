import numpy as np

from src.perception.temporal_event_classifier_training import classification_metrics, train_classifier
from src.perception.temporal_event_classifier_model import predict_water_probability


CONFIG = {"learning_rate": .1, "l2_regularization": .01, "iterations": 500,
          "class_weight_water": 2, "class_weight_dry": 1}


def _sample(value, label, index):
    return {"feature_vector": np.asarray([value, value * .5]), "label": label,
            "sequence": {"path": f"s{index // 4}", "rain_level": "moderate"}}


def test_numpy_logistic_model_learns_separable_features():
    samples = [_sample(-2 - i * .1, "background_noise", i) for i in range(8)]
    samples += [_sample(2 + i * .1, "water_ripple", i + 8) for i in range(8)]
    model = train_classifier(samples, CONFIG)
    probability, _ = predict_water_probability(np.stack([item["feature_vector"] for item in samples]), model)
    metrics = classification_metrics(samples, probability, .4, .6)
    assert metrics["water_ripple"]["f1"] > .9
    assert metrics["dry_or_noise"]["f1"] > .9
