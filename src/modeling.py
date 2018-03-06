import vg_pricing as vg
import heston_pricing as he
import ls_pricing as ls
import black_scholes as bs
import numpy as np
import scipy.optimize as opt
from structs import EvalArgs
from pars_range import ParsRange
import optimization
import data_helpers as dh

par_bounds = {
    "heston": ((1e-6, 45), (1e-7, 1), (1e-7, 2), (-1, 1), (1e-10, 1)),
    "vg":     ((1e-6, 3), (-2, 2), (1e-6, 2)),
    "ls":     ((1.00001, 1.99999), (1e-6, 2)),
    "bs":     ((1e-10, 10), )
}

models = {
    "heston": he.price_heston,
    "vg":     vg.price_vg,
    "ls":     ls.price_ls,
    "bs":     bs.price_bs
}


def model_prices(pars: tuple, args: EvalArgs, model: str,
                 strict=False, check=False, bounds_only=True) -> np.ndarray:
    return models[model](pars=pars, args=args.as_tuple(),
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
    bounds2 = par_bounds[model2]

    for pars1 in ParsRange(model=model1, center=center, widths=widths, dots=dots):
        prices = model_prices(pars=pars1, args=args, model=model1)
        result = tune_model(eval_args=args, bounds=bounds2, metric=metric, model=model2, prices=prices, local=False)
        pars2 = result.x

        with open(f"params/{model1}_{model2}_{metric}.txt", "a") as out:
            out.write(f"{dh.array2str(pars1)} --> {dh.array2str(pars2)} with quality metric {metric}: {result.fun}\n")

        print(f"Estimated for {dh.array2str(pars1)}")

    print("Done")
