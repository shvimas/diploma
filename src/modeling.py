import numpy as np
import scipy.optimize as opt

import config
import data_helpers as dh
import optimization
from pars_range import ParsRange
from structs import EvalArgs


def model_prices(pars: tuple, args: EvalArgs, model: str,
                 strict=False, check=False, bounds_only=True) -> np.ndarray:
    return optimization.models[model](pars=pars, args=args.as_tuple(),
                                      strict=strict, check=check, bounds_only=bounds_only)


def tune_model(eval_args: EvalArgs, bounds: tuple, model: str, metric: str, prices: np.ndarray,
               local=False, **kwargs) -> opt.OptimizeResult:
    """
    Finds best(global or local) parameters for specified model

    :param eval_args: market parameters + quality metric as args[-2] + actual prices as args[-1]
    :param bounds: model parameters natural bounds in format: ((min1, max1), ...)
    :param model: pricing model
    :param metric: quality metric
    :param prices: actual prices
    :param local: should use local search?

    :return: optimal parameters
    """

    if local:
        try:
            kwargs.pop('disp')
        except KeyError:
            pass
        try:
            kwargs.pop('maxiter')
        except KeyError:
            pass
        try:
            kwargs.pop('polish')
        except KeyError:
            pass
        res = opt.minimize(
                fun=lambda pars: optimization.estimate_model(pars, eval_args, model, metric, prices),
                bounds=bounds,
                **kwargs)
    else:
        try:
            kwargs.pop('x0')
        except KeyError:
            pass
        res = opt.differential_evolution(
                func=lambda pars: optimization.estimate_model(pars, eval_args, model, metric, prices),
                bounds=bounds,
                **kwargs)

    return res


def tune_on_near_params(model1: str, model2: str, args: EvalArgs, metric: str,
                        center: tuple, widths: tuple, dots: int):
    bounds2 = config.par_bounds[model2]

    for pars1 in ParsRange(model=model1, center=center, widths=widths, dots=dots):
        prices = model_prices(pars=pars1, args=args, model=model1)
        result = tune_model(eval_args=args, bounds=bounds2, metric=metric, model=model2, prices=prices, local=False)
        pars2 = result.x

        with open(f"params/{model1}_{model2}_{metric}.txt", "a") as out:
            out.write(f"{dh.array2str(pars1)} --> {dh.array2str(pars2)} with quality metric {metric}: {result.fun}\n")

        print(f"Estimated for {dh.array2str(pars1)}")

    print("Done")
