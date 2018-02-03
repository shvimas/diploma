from data_helpers import read_data, array2str
from optimization import estimate_model, is_good_enough, metrics
from Heston_Pricing_Integral_vectorized import price_heston
import scipy.optimize as opt
import numpy as np
from modeling import par_bounds
from structs import Info, Data, EvalArgs
from modeling import tune_on_near_params
from time import time
from datetime import timedelta
from data_helpers import prepare_data
from typing import List


def opt_func(pars, *args) -> float:
    if len(args) != 6:
        raise ValueError("args must have exactly 6 items")
    out = args[0]
    log2console = args[1]
    model = args[2]
    metric = args[3]
    actual = args[4]
    quality = estimate_model(pars, EvalArgs.from_tuple(args[5]), model, metric, actual)

    msg = metric + ": " + str(quality) + " with params: " + array2str(pars)
    if log2console:
        print(msg)

    if is_good_enough(quality, metric):
        out.write(msg + "\n")

    return quality


def optimize_model(model: str, info: List[Info], data: Data,
                   metric: str, day: int, is_call: bool, rate: float,
                   log2console, disp=False) -> opt.OptimizeResult:
    """
    :param model:
    :param info:
    :param data:
    :param metric:
    :param day:
    :param is_call:
    :param rate:
    :param log2console:
    :param disp: display status messages in diff evolution
    :return: opt.OptimizeResult
    """

    print(f"Optimizing {model} with {metric} on day {day}")

    with open(f"params/{model}_{metric}_good_params.txt", "a") as good:
        good.write("Day: " + str(day) + "\n")

        strikes = data.strikes[is_call][day]
        actual = data.prices[is_call][day]

        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        args = (good, log2console, model, metric, actual, (spot, strikes, maturity, rate, q, is_call))
        bounds = par_bounds[model]

        t0 = time()
        best_pars = opt.differential_evolution(func=opt_func, bounds=bounds, disp=disp, args=args)
        print(f"Time spent: {str(timedelta(seconds=(time() - t0)))}\n")

        good.write("\n")

    return best_pars


def tune_all_models(args: EvalArgs):
    """
    :param args:
    :return:
    """

    '''
    all_models = {"heston", "vg", "ls"}
    for model1 in all_models:
        for model2 in all_models - {model1}:
            pass
    '''

    center = (.05, -.15, .04)
    widths = (.01, .03, .01)
    tune_on_near_params(model1="vg", model2="ls", args=args, metric="RMR", center=center, widths=widths, dots=100)


def main() -> None:
    data, info = read_data("SPH2_031612.csv")

    # pars_heston = (5.73144671461, 0.00310912079833, 0.200295855838, 0.0131541339298, 0.0295404046434)
    # pars_heston = (0.405, 0.0098, 0.505, 0.00057, 0.04007)
    pars_heston = (4.2216740989, 0.0199176675743, 1.51769128617e-05, 0.0474806534178, 0.000569295223402)
    # pars_vg = (0.996575637472, -0.142224286732, 0.0954970105615)
    # pars_vg = (0.999728271222, -0.124716144066, 0.109217167741)

    day = 0

    market = EvalArgs(spot=info[day].spot, k=data.strikes[True][day], tau=info[day].mat, r=.03, q=.03, call=True)

    '''
    from rate import find_opt_rates
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = False
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = True
    '''

    print(metrics["RMR"](price_heston(pars=pars_heston, args=market.as_tuple()), data.prices[market.is_call][day]))

    # tune_all_models(market)

    log2console = False
    metric = "RMR"
    heston_best = open(f"params/best4heston_{metric}.txt", "w")
    vg_best = open(f"params/best4vg_{metric}.txt", "w")
    ls_best = open(f"params/best4ls_{metric}.txt", "w")

    # need to somehow work around with overflows
    np.seterr(all='warn')

    try:
        data, info = prepare_data(data=data, info=info)

        for day in range(0, len(info)):
            for model_name, file in [("heston", heston_best), ("vg", vg_best), ("ls", ls_best)]:
                p1 = optimize_model(model=model_name, info=info, data=data,
                                    metric=metric, day=day, rate=.03,
                                    is_call=True, log2console=log2console, disp=True)
                file.write(f"Day {day} with func value {p1.fun}: {array2str(p1.x)}\n")
                file.flush()
    finally:
        heston_best.close()
        vg_best.close()
        ls_best.close()


if __name__ == "__main__":
    main()
