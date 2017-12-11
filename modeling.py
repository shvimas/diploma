from VG_Pricing_Integral_vectorized import price_vg
from Heston_Pricing_Integral_vectorized import price_heston
from Log_Stable_Pricing import price_ls
import numpy as np
import scipy.optimize as opt
from eval_args import EvalArgs
from optimization import estimate_model


models = {
    "heston": price_heston,
    "vg": price_vg,
    "ls": price_ls
}


def model_prices(pars: tuple, args: EvalArgs, model: str) -> np.ndarray:
    return models[model](pars=pars, args=args.as_tuple())


def tune_model(args: EvalArgs,
               bounds: tuple,
               model: str,
               metric: str,
               prices: np.ndarray,
               local=False,
               **kwargs) -> np.ndarray:
    """
    Finds best(global or local) parameters for specified model

    :param args: market parameters + quality metric as args[-2] + actual prices as args[-1]
    :param bounds: model parameters natural bounds in format: ((min1, max1), ...)
    :param model: pricing model
    :param metric: quality metric
    :param prices: actual prices
    :param local: should use local search?

    :return: optimal parameters
    """

    if local:
        res = opt.minimize(
            fun=lambda pars: estimate_model(pars, args, model, metric, prices),
            bounds=bounds,
            **kwargs
        )
    else:
        res = opt.differential_evolution(
            func=lambda pars: estimate_model(pars, args, model, metric, prices),
            bounds=bounds,
            **kwargs
        )

    return res.x
