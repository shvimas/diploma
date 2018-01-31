from data_helpers import read_data, array2str
from optimization import estimate_model, is_good_enough, metrics
from Heston_Pricing_Integral_vectorized import price_heston
import scipy.optimize as opt
import numpy as np
from eval_args import EvalArgs
from modeling import par_bounds
from structs import Info, Data
from typing import List
from math import exp
from modeling import tune_on_near_params


def remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info: List[Info], rate=.03):
    for day in range(len(info)):
        spot = info[day].spot
        tau = info[day].mat / len(info)
        otm_call = strikes_call[day] > spot * exp(rate * tau)
        otm_put = strikes_put[day] < spot

        strikes_call[day] = strikes_call[day][otm_call]
        prices_call[day] = prices_call[day][otm_call]
        strikes_put[day] = strikes_put[day][otm_put]
        prices_put[day] = prices_put[day][otm_put]

    return strikes_call, strikes_put, prices_call, prices_put, info


def cut_tails(strikes_call, strikes_put, prices_call, prices_put, info, min_perc=.01, min_price=10):
    for day in range(len(info)):
        spot = info[day].spot
        good_calls = np.logical_and(prices_call[day] > min_price, prices_call[day] / spot > min_perc)
        good_puts = np.logical_and(prices_put[day] > min_price, prices_put[day] / spot > min_perc)
        strikes_call[day] = strikes_call[day][good_calls]
        prices_call[day] = prices_call[day][good_calls]
        strikes_put[day] = strikes_put[day][good_puts]
        prices_put[day] = prices_put[day][good_puts]

    return strikes_call, strikes_put, prices_call, prices_put, info


def prepare_data(info: List[Info], strikes_call: List[np.ndarray], strikes_put: List[np.ndarray],
                 prices_call: List[np.ndarray], prices_put: List[np.ndarray]):
    return remove_itm_options(
            *cut_tails(strikes_call=strikes_call, strikes_put=strikes_put,
                       prices_call=prices_call, prices_put=prices_put, info=info))


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


def optimize_model(model: str, info: list,
                   strikes_call: list, strikes_put: list,
                   prices_call: list, prices_put: list,
                   metric: str, day: int, is_call: bool, rate: float,
                   log2console, disp=False) -> opt.OptimizeResult:
    """
    :param model:
    :param info:
    :param strikes_call:
    :param strikes_put:
    :param prices_call:
    :param prices_put:
    :param metric:
    :param day:
    :param is_call:
    :param rate:
    :param log2console:
    :param disp: display status messages in diff evolution
    :return:
    """

    print(f"Optimizing {model} with " + metric + " on day " + str(day))

    with open(f"params/{model}_" + metric + "_good_params.txt", "a") as good:
        good.write("Day: " + str(day) + "\n")
        if is_call:
            strikes = strikes_call[day]
            actual = prices_call[day]
        else:
            strikes = strikes_put[day]
            actual = prices_put[day]

        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        args = (good, log2console, model, metric, actual, (spot, strikes, maturity, rate, q, is_call))
        bounds = par_bounds[model]

        best_pars = opt.differential_evolution(func=opt_func, bounds=bounds, disp=disp, args=args)

    return best_pars


def tune_all_models(args: EvalArgs):
    '''
    all_models = {"heston", "vg", "ls"}
    for model1 in all_models:
        for model2 in all_models - {model1}:
            pass
    '''

    center = (.05, -.15, .04)
    widths = (.01, .03, .01)
    tune_on_near_params(model1="vg", model2="ls", args=args, metric="RMR", center=center, widths=widths, dots=100)


# noinspection PyMissingTypeHints
def main():
    info, strikes_call, strikes_put, prices_call, prices_put = read_data("SPH2_031612.csv")

    # pars_heston = (5.73144671461, 0.00310912079833, 0.200295855838, 0.0131541339298, 0.0295404046434)
    # pars_heston = (0.405, 0.0098, 0.505, 0.00057, 0.04007)
    pars_heston = (4.2216740989, 0.0199176675743, 1.51769128617e-05, 0.0474806534178, 0.000569295223402)
    # pars_vg = (0.996575637472, -0.142224286732, 0.0954970105615)
    # pars_vg = (0.999728271222, -0.124716144066, 0.109217167741)

    day = 0

    market = EvalArgs(spot=info[day].spot, k=strikes_call[day], tau=info[day].mat, r=.03, q=.03, call=True)

    '''
    from rate import find_opt_rates
    find_opt_rates(args=market, actual=prices_call[day])
    market.is_call = False
    find_opt_rates(args=market, actual=prices_put[day])
    market.is_call = True
    '''

    print(metrics["RMR"](price_heston(pars=pars_heston, args=market.as_tuple()), prices_call[day]))

    # tune_all_models(market)

    log2console = False
    metric = "RMR"
    heston_best = open(f"params/best4heston_{metric}.txt", "w")
    vg_best = open(f"params/best4vg_{metric}.txt", "w")
    ls_best = open(f"params/best4ls_{metric}.txt", "w")

    strikes_call, strikes_put, prices_call, prices_put, info = \
        prepare_data(strikes_call=strikes_call, strikes_put=strikes_put,
                     prices_call=prices_call, prices_put=prices_put, info=info)

    for day in range(0, len(info)):
        for model_name, file in [("heston", heston_best), ("vg", vg_best), ("ls", ls_best)]:
            p1 = optimize_model(model=model_name, info=info, strikes_call=strikes_call, strikes_put=strikes_put,
                                prices_call=prices_call, prices_put=prices_put,
                                metric=metric, day=day, rate=.03, is_call=True, log2console=log2console)
            file.write(f"Day {day} with func value {p1.fun}: {array2str(p1.x)}\n")
            file.flush()

    heston_best.close()
    vg_best.close()
    ls_best.close()


if __name__ == "__main__":
    main()
