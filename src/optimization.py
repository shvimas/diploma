import numpy as np
from structs import EvalArgs
import modeling as mo
import config


def mean(seq) -> float:
    return float(np.sum(seq) / len(seq))


def mean_absolute_error(predicted, actual) -> float:
    return mean(np.abs(predicted - actual))


def ratio(predicted: np.ndarray, actual: np.ndarray) -> list:
    for i in range(len(predicted)):
        if abs(predicted[i]) < config.eps:
            predicted[i] = config.eps
    return list(map(lambda x: np.abs(x) if x >= 1 else np.abs(1 / x), predicted / actual))


def mean_ratio(predicted, actual) -> float:
    return mean(ratio(predicted, actual))


def robust_mean_ratio(predicted, actual, alpha=.05) -> float:
    tmp = ratio(predicted, actual)
    n = len(tmp)
    p = n - int(n * alpha)
    return mean(sorted(tmp)[:p])


metrics = {
    "MAE":        mean_absolute_error,
    "mean ratio": mean_ratio,
    "MAR":        mean_ratio,
    "RMR":        robust_mean_ratio
}


def apply_metric(predicted, actual, metric="MAE", **kwargs) -> float:
    try:
        return metrics[metric](predicted, actual, **kwargs)
    except Warning:
        return config.inf_metric


def is_good_enough(quality: float, metric: str) -> bool:
    values = {
        "MAE":        2,
        "mean ratio": 1.02,
        "RMR":        1.02
    }

    return quality < values[metric]


def estimate_model(pars: tuple, args: EvalArgs, model: str, metric: str, prices: np.ndarray) -> float:
    """

    :param pars: model parameters
    :param args: market parameters with strikes as args[1]
    :param model: pricing model
    :param metric: quality metric
    :param prices: actual prices

    :return: value of quality metric on passed parameters
    """

    k = args.get_strikes()

    if model not in mo.models.keys():
        raise Exception("Cannot use model " + model)

    if metric not in metrics.keys():
        raise Exception("Cannot use metric " + metric)

    if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
        raise Exception("strikes and prices should be np.arrays with same length")

    return apply_metric(mo.models[model](pars=pars, args=args.as_tuple()), prices, metric)
