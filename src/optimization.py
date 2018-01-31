import numpy as np
from structs import EvalArgs
from VG_Pricing_Integral_vectorized import price_vg
from Heston_Pricing_Integral_vectorized import price_heston
from Log_Stable_Pricing import price_ls


def mean(seq) -> float:
    return float(np.sum(seq) / len(seq))


def mean_absolute_error(predicted, actual) -> float:
    return mean(abs(predicted - actual))


'''
def ratio(predicted: np.ndarray, actual: np.ndarray, max_val=1e10) -> np.ndarray:
    for i in range(len(predicted)):
        if predicted[i] == np.Inf:
            predicted[i] = max_val
    tmp = list(map(lambda x: abs(x) if x >= 1 else abs(1 / x), predicted / actual))
    # finish this
'''


def mean_ratio(predicted, actual) -> float:
    tmp = list(map(lambda x: abs(x) if x >= 1 else abs(1 / x), predicted / actual))
    return mean(tmp)


def robust_mean_ratio(predicted, actual, alpha=.05) -> float:
    tmp = list(map(lambda x: abs(x) if x >= 1 else abs(1 / x), predicted / actual))
    n = len(tmp)
    p = n - int(n * alpha)
    return mean(sorted(tmp)[:p])


models = {
    "heston": price_heston,
    "vg":     price_vg,
    "ls":     price_ls
}

metrics = {
    "MAE":        mean_absolute_error,
    "mean ratio": mean_ratio,
    "RMR":        robust_mean_ratio
}


def apply_metric(predicted, actual, metric="MAE") -> float:
    return metrics[metric](predicted, actual)


def is_good_enough(quality: float, metric: str) -> bool:
    values = {
        "MAE":        2,
        "mean ratio": 1.05,
        "RMR":        1.04
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

    k = args.strikes

    if model not in models.keys():
        raise Exception("Cannot use model " + model)

    if metric not in metrics.keys():
        raise Exception("Cannot use metric " + metric)

    if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
        raise Exception("strikes and prices should be np.arrays with same length")

    return apply_metric(models[model](pars=pars, args=args.as_tuple()), prices, metric)
