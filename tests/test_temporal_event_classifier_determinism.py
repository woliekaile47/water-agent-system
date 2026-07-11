import numpy as np

from src.perception.temporal_event_classifier_model import fit_logistic_regression


def test_training_is_bitwise_deterministic():
    x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    config = {"learning_rate": .1, "l2_regularization": .01, "iterations": 200}
    first = fit_logistic_regression(x, y, weights, config)
    second = fit_logistic_regression(x, y, weights, config)
    assert np.array_equal(first["weights"], second["weights"])
    assert first["bias"] == second["bias"]
