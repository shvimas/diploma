from data_helpers import read_data, array2str, get_filename, prepare_data, extract_centers
from optimization import estimate_model, is_good_enough
import scipy.optimize as opt
from modeling import par_bounds
from structs import Info, Data, EvalArgs
from modeling import tune_on_near_params
from time import time
from datetime import timedelta
from typing import List
from sklearn.decomposition import PCA
from multiprocessing import Pool
import re


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

    with open(get_filename(model=model, metric=metric, best=False, is_call=is_call), "a") as good:
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
        print(f"Time spent for {model}, day {day}: {str(timedelta(seconds=(time() - t0)))}\n")

        good.write("\n")

    return best_pars


def tune_all_models(args: EvalArgs, metric: str):

    def get_centers(model: str, metric_: str, is_call: bool) -> str:
        return extract_centers(get_filename(model=model, metric=metric_, is_call=is_call))

    models = {"heston", "vg", "ls"}
    for model1, model2, is_call in [(m1, m2, c) for m1 in models for m2 in models - {m1} for c in [True, False]]:
        centers1 = get_centers(model=model1, metric_=metric, is_call=is_call)
        pca = PCA(n_components=2)
        factors = pca.components_
        centers1_2d = pca.fit_transform(centers1)



    center = (.05, -.15, .04)
    widths = (.01, .03, .01)
    tune_on_near_params(model1="vg", model2="ls", args=args, metric="RMR", center=center, widths=widths, dots=100)


def get_last_day(filename: str) -> int:
    try:
        with open(filename) as f:
            return int(re.search(r'Day (.*?)[:\s]', f.readlines()[-1]).group(1))
    except FileNotFoundError or IndexError:
        return -1


def func(args: tuple):
    model = args[0]
    metric = args[1]
    info = args[2]
    kwargs = args[3]
    start = get_last_day(get_filename(model=model, metric=metric, is_call=kwargs['is_call']))
    file = open(get_filename(model=model, metric=metric, is_call=kwargs['is_call']), 'a')
    for day in range(start + 1, len(info)):
        p1 = optimize_model(model=model, info=info, metric=metric, day=day, **kwargs)
        file.write(f"Day {day} with func value {p1.fun}: {array2str(p1.x)}\n")
        file.flush()
    file.close()


def main() -> None:
    data, info = read_data("SPH2_031612.csv")

    day = 0

    market = EvalArgs(spot=info[day].spot, k=data.strikes[True][day], tau=info[day].mat, r=.03, q=.03, call=True)

    '''
    from rate import find_opt_rates
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = False
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = True
    '''

    # tune_all_models(market, "RMR")

    log2console = False
    metric = "MAE"

    data, info = prepare_data(data=data, info=info)

    pool = Pool()
    models = ('heston', )
    kwargs = [{
        'data': data,
        'rate': .03,
        'is_call': True,
        'log2console': log2console,
        'disp': False
    }] * len(models)
    all_args = zip(models, [metric] * len(models), [info] * len(models), kwargs)
    pool.map(func, all_args)

    '''
    heston_best = open(get_filename(model='heston', metric=metric), "w")
    vg_best = open(get_filename(model='vg', metric=metric), "w")
    ls_best = open(get_filename(model='ls', metric=metric), "w")

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
    '''


if __name__ == "__main__":
    main()
