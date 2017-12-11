import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error
from modeling import models
from eval_args import EvalArgs


def mean(seq):
    return sum(seq) / len(seq)


def mean_ratio(predicted, actual):
    tmp = list(map(
        lambda x: (x >= 1) * x + (x < 1) / x,
        predicted / actual
    ))

    return mean(tmp)


metrics = {
        "MAE": mean_absolute_error,
        "mean ratio": mean_ratio
    }


def apply_metric(predicted, actual, metric="MAE"):
    return metrics[metric](predicted, actual)


def is_good_enough(quality: float, metric: str) -> bool:
    values = {
        "MAE": 2,
        "mean ratio": 1.05
    }

    return quality < values[metric]


def estimate_model(pars: tuple, args: EvalArgs, model: str, metric: str, prices: np.ndarray):
    """

    :param pars: model parameters
    :param args: market parameters with strikes as args[1]
    :param model: pricing model
    :param metric: quality metric
    :param prices: actual prices

    :return: value of quality metric on passed parameters
    """

    k = args.strikes

    if model not in models.keys():
        raise Exception("Cannot use model " + model)

    if metric not in metrics.keys():
        raise Exception("Cannot use metric " + metric)

    if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
        raise Exception("strikes and prices should be np.arrays with same length")

    return apply_metric(models[model](pars=pars, args=args.as_tuple()), prices, metric)
