from VG_Pricing_Integral_vectorized import price_vg
from Heston_Pricing_Integral_vectorized import price_heston
from Log_Stable_Pricing import price_ls
import numpy as np
import scipy.optimize as opt
from structs import EvalArgs
from pars_range import ParsRange
from optimization import estimate_model
from data_helpers import array2str

par_bounds = {
    "heston": ((1e-6, 10), (1e-7, 1), (1e-7, 2), (0, 1), (1e-10, 1)),
    "vg":     ((1e-6, 1), (-1, 1), (1e-6, 1)),
    "ls":     ((1.00001, 1.99999), (1e-6, 2))
}

models = {
    "heston": price_heston,
    "vg":     price_vg,
    "ls":     price_ls
}


def model_prices(pars: tuple, args: EvalArgs, model: str) -> np.ndarray:
    return models[model](pars=pars, args=args.as_tuple())


def tune_model(args: EvalArgs, bounds: tuple, model: str, metric: str, prices: np.ndarray,
               local=False, **kwargs) -> opt.OptimizeResult:
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
                **kwargs)
    else:
        res = opt.differential_evolution(
                func=lambda pars: estimate_model(pars, args, model, metric, prices),
                bounds=bounds,
                **kwargs)

    return res


def tune_on_near_params(model1: str, model2: str, args: EvalArgs, metric: str,
                        center: tuple, widths: tuple, dots: int):
    bounds2 = par_bounds[model2]

    for pars1 in ParsRange(model=model1, center=center, widths=widths, dots=dots):
        prices = model_prices(pars=pars1, args=args, model=model1)
        result = tune_model(args=args, bounds=bounds2, metric=metric, model=model2, prices=prices, local=False)
        pars2 = result.x

        with open("params/" + model1 + "_" + model2 + "_" + metric + ".txt", "a") as out:
            out.write(array2str(pars1) + " --> " + array2str(pars2) +
                      " with quality metric " + metric + ": " + str(result.fun) + "\n")

        print(f"Estimated for {array2str(pars1)}")

    print("Done")
