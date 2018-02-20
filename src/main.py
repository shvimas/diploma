from optimization import estimate_model, is_good_enough
import scipy.optimize as opt
from modeling import par_bounds
from structs import Info, Data, EvalArgs
from modeling import tune_model, model_prices
from time import time
from datetime import timedelta
from typing import List
from multiprocessing import Pool
import re
import numpy as np
import data_helpers as dh


def opt_func(pars, *args) -> float:
    if len(args) != 6:
        raise ValueError("args must have exactly 6 items")
    out = args[0]
    log2console = args[1]
    model = args[2]
    metric = args[3]
    actual = args[4]
    quality = estimate_model(pars, EvalArgs.from_tuple(args[5]), model, metric, actual)

    msg = metric + ": " + str(quality) + " with params: " + dh.array2str(pars)
    if log2console:
        print(msg)

    if is_good_enough(quality, metric):
        out.write(msg + "\n")

    return quality


def optimize_model(model: str, info: List[Info], data: Data,
                   metric: str, day: int, is_call: bool, rate: float,
                   local: bool, **kwargs) -> opt.OptimizeResult:

    print(f"Optimizing {model} with {metric} on day {day}")

    with open(dh.get_filename(model=model, metric=metric, best=False, is_call=is_call), "a") as good:
        good.write("Day: " + str(day) + "\n")

        strikes = data.strikes[is_call][day]
        actual = data.prices[is_call][day]

        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        t0 = time()

        best_pars = tune_model(eval_args=EvalArgs.from_tuple((spot, strikes, maturity, rate, q, is_call)),
                               bounds=par_bounds[model], model=model, metric=metric,
                               prices=actual, local=local, **kwargs)
        print(f"Time spent for {model}, day {day}: {timedelta(seconds=(time() - t0))}\n")

        good.write("\n")

    return best_pars


def tune_all_models(args: EvalArgs, metric: str):
    models = {"heston", "vg", "ls"}
    for model1, model2, is_call in [(m1, m2, c) for m1 in models for m2 in models - {m1} for c in [True, False]]:
        print(f"Tuning {model2} with {model1}")
        args.is_call = is_call
        bounds, factors, means = dh.get_pca_data(model=model1, is_call=is_call)
        grid = dh.grid(*bounds, n=20)  # 20 dots for each dimension => 400 dots, but many will not be priced
        with open(f'params/tune_{model1}_{model2}_{"call" if is_call else "put"}.txt', 'w') as f:
            for dot in grid:
                pars = dot @ factors + means
                try:
                    prices = model_prices(pars=pars, args=args, model=model1,
                                          strict=True, check=True, bounds_only=True)
                    print(f"{dh.array2str(dot)} -> {dh.array2str(pars)}")
                    if sum(prices > 1e5) > 0:
                        raise ValueError(f"overflow in prices: {prices}")

                    t0 = time()
                    res = tune_model(eval_args=args, bounds=par_bounds[model2], model=model2, metric=metric,
                                     prices=prices, local=False, polish=True, maxiter=25)

                    f.write(f"Pars {dh.array2str(pars)} from dot {dh.array2str(dot)} "
                            f"with metric {metric} = {res.fun}: {dh.array2str(res.x)}\n")
                    f.flush()
                    print(f"Metric={res.fun} with {model2} params: {res.x}")
                    print(f"Time spent: {timedelta(seconds=(time() - t0))}\n")
                except ValueError as e:
                    print(f'Skipped because {e}')

        print("Done\n")


def get_last_day(filename: str) -> int:
    try:
        with open(filename) as f:
            return int(re.search(r'Day (.*?)[:\s]', f.readlines()[-1]).group(1))
    except:  # FileNotFoundError or IndexError:
        return -1


def func(args: tuple):
    model = args[0]
    metric = args[1]
    info = args[2]
    cycle = args[3]
    kwargs = args[4]
    start = get_last_day(dh.get_filename(model=model, metric=metric, is_call=kwargs['is_call']))
    file = open(dh.get_filename(model=model, metric=metric, is_call=kwargs['is_call']), 'a')
    prev = None
    local = False
    for day in range(start + 1, len(info)):
        if model == 'heston' or model == 'vg':
            local = day % cycle != 0 and prev is not None
        if local:
            kwargs['x0'] = prev.x
        p1 = optimize_model(model=model, info=info, metric=metric, day=day, local=local, **kwargs)
        file.write(f"Day {day} with func value {p1.fun}: {dh.array2str(p1.x)}\n")
        file.flush()
        prev = p1
    file.close()


def main() -> None:

    # need to somehow work around with overflows
    np.seterr(all='warn')

    data, info = dh.read_data("SPH2_031612.csv")

    day = 0
    market = EvalArgs(spot=info[day].spot, k=data.strikes[True][day],
                      tau=info[day].mat / len(info),
                      r=.03, q=.03, call=True)

    tune_all_models(market, "MAE")
    '''
    from rate import find_opt_rates

    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = False
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = True
    

    log2console = False
    metric = "MAE"

    data, info = dh.prepare_data(data=data, info=info)

    pool = Pool()
    models = ('heston', 'vg', 'ls')
    kwargs = [{
        'data': data,
        'rate': .03,
        'is_call': True,
        'log2console': log2console,
        'disp': False,
        'polish': True,
        'maxiter': 50
    }] * len(models)
    all_args = zip(models, [metric] * len(models), [info] * len(models), [10] * len(models), kwargs)
    pool.map(func, all_args)
    '''


if __name__ == "__main__":
    main()
