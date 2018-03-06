import helper_funcs
import scipy.optimize as opt
from modeling import par_bounds
from structs import Info, Data, EvalArgs
from time import time
from datetime import timedelta
from typing import List
from multiprocessing import Pool
import re
import numpy as np
import data_helpers as dh
from gen_pricer import GenPricer


def optimize_model(model: str, info: List[Info], data: Data,
                   metric: str, day: int, rate: float,
                   local: bool, use_fft: bool, **kwargs) -> opt.OptimizeResult:

    print(f"Optimizing {model} with {metric} on day {day}")

    actual_calls = data.prices[True][day]
    actual_puts = data.prices[False][day]
    pricer = GenPricer(model=model,
                       market=EvalArgs.from_structure(data=data, info=info, rate=rate, day=day),
                       use_fft=use_fft)
    optimizer = opt.minimize if local else opt.differential_evolution

    t0 = time()
    result = pricer.optimize_pars(metric=metric,
                                  actual_calls=actual_calls,
                                  actual_puts=actual_puts,
                                  bounds=par_bounds[model],
                                  optimizer=optimizer,
                                  **kwargs)
    print(f"Time spent for {model}, day {day}: {timedelta(seconds=(time() - t0))}\n")

    return result


def tune_all_models(args: EvalArgs, metric: str):
    args.is_call = None
    model1 = 'vg'
    models2 = {"heston", "ls"}
    for model2 in models2:
        pricer1 = GenPricer(model=model1, market=args, use_fft=True)
        pricer2 = GenPricer(model=model2, market=args, use_fft=True)
        print(f"Tuning {model2} with {model1}")
        bounds, factors, means = dh.get_pca_data(model=model1)
        grid = helper_funcs.grid(*bounds, n=20)  # 20 dots for each dimension => 400 dots, but some will not be priced
        starting_dot = np.array([-np.Inf, -np.Inf])
        with open(f'params/tune_{model2}_with_{model1}.txt', 'a+') as f:
            try:
                f.seek(0)
                lines = f.readlines()
                str_dot = re.search(r'from dot (.+?) with metric', lines[-1]).group(1)
                starting_dot = np.array(helper_funcs.extract_floats(str_dot))
            except IndexError or AttributeError:
                pass
            for dot in grid:
                if not np.all(dot[0] > starting_dot[0] or (dot[0] == starting_dot[0] and dot[1] > starting_dot[1])):
                    print(f"Skipped dot {helper_funcs.array2str(dot)} -- already evaluated")
                    continue
                pars = dot @ factors + means
                if dh.bad_pars(pars=pars, bounds_only=True, model=model1):
                    print(f"Skipped dot {dot} because of bad {model1} pars {pars}")
                    continue
                try:
                    call_prices, put_prices = pricer1.price(pars=pars)
                    print(f"{dh.array2str(dot)} -> {dh.array2str(pars)}")
                    if sum(call_prices > 1e5) > 0:
                        raise ValueError(f"overflow in call prices: {call_prices}")
                    if sum(put_prices > 1e5) > 0:
                        raise ValueError(f"overflow in put prices: {put_prices}")

                    t0 = time()
                    res = pricer2.optimize_pars(metric=metric, bounds=par_bounds[model2],
                                                actual_puts=put_prices,
                                                actual_calls=call_prices,
                                                optimizer=opt.differential_evolution,
                                                polish=True, maxiter=50, disp=False)

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


def calibrate(args: tuple):
    assert len(args) == 4
    model = args[0]
    metric = args[1]
    info = args[2]
    kwargs = args[3]
    is_call = kwargs.pop('is_call')
    start = get_last_day(helper_funcs.get_filename(model=model, metric=metric, is_call=is_call))
    file = open(helper_funcs.get_filename(model=model, metric=metric, is_call=is_call), 'a')
    for day in range(start + 1, len(info)):
        result = optimize_model(model=model, info=info, metric=metric, day=day, local=False, **kwargs)
        file.write(f"Day {day} with func value {result.fun}: {dh.array2str(result.x)}\n")
        file.flush()
    file.close()


def main() -> None:

    # need to somehow work around with overflows
    np.seterr(all='warn')

    data, info = helper_funcs.read_data("SPH2_031612.csv")

    day = 0
    market = EvalArgs(spot=info[day].spot,
                      k_call=data.strikes[True][day],
                      k_put=data.strikes[False][day],
                      tau=info[day].mat, r=.008, q=.008, call=None)

    tune_all_models(market, "MAE")
    '''
    from rate import find_opt_rates

    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = False
    find_opt_rates(args=market, actual=data.prices[market.is_call][day])
    market.is_call = True
    

    metric = "MAE"
    models = ('vg', )
    kwargs = {
        'data': data,
        'rate': .008,
        'is_call': None,  # both puts and calls
        'disp': False,
        'polish': True,
        'maxiter': 100,
        'use_fft': True
    }

    all_args = zip(models, [metric] * len(models), [info] * len(models), [kwargs] * len(models))

    pool = Pool()
    pool.map(calibrate, all_args)
    '''



if __name__ == "__main__":
    main()
