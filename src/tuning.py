import os
from datetime import timedelta
from time import time
from typing import List

import numpy as np
from scipy import optimize as opt

import data_helpers as dh
import helper_funcs as hf
from config import par_bounds
from gen_pricer import GenPricer
from structs import Info, Data, EvalArgs


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
    with open(hf.get_log_file_name(model=model, metric=metric), 'a+') as log:
        hf.log_print(f"Time spent for {model}, day {day}: {timedelta(seconds=(time() - t0))}\n", out1=log)

    return result


# noinspection PyBroadException
def tune_model(args: tuple):
    """
    model1: model to be priced for tuning model2
    model3: model with which model1 was tuned

    :param args: model1, model2, model3, market, metric, logfile, from_grid
    """
    if len(args) != 7:
        raise ValueError('expected exactly 7 arguments: model1, model2, model3, market, metric, logfile, from_grid')
    model1, model2, model3, market, metric, logfile, from_grid = args

    if os.path.isfile(hf.get_flagfile_name(pricing_model=model1, tuning_model=model2, metric=metric)):
        return

    if not from_grid and not os.path.isfile(
            hf.get_flagfile_name(pricing_model=model3, tuning_model=model1, metric=metric)):
        raise ValueError(f"{model1} is not yet tuned with {model3}")

    if market.is_call is not None:
        raise ValueError('market.is_call is supposed to None(means both calls and puts)')

    if not logfile.startswith('params/') and os.getcwd().endswith('/diploma'):
        logfile = 'params/' + logfile

    with open(logfile, 'a+') as log, open(hf.get_tune_file_name(
            pricing_model=model1,
            tuning_model=model2,
            metric=metric), 'a+') as f:
        pricer1 = GenPricer(model=model1, market=market, use_fft=True)
        pricer2 = GenPricer(model=model2, market=market, use_fft=True)
        hf.log_print(f"\nTuning {model2} with {model1}", log)

        dots = hf.get_tuning_dots(pricing_model=model1 if from_grid else model3,
                                  tuning_model=model1,
                                  metric=metric,
                                  from_grid=from_grid)

        starting_dot = None
        skip = False
        try:
            starting_dot = hf.get_starting_dot(f, from_grid)
            skip = True
        except IndexError or AttributeError:
            pass

        for pars in dots:
            if skip:
                hf.log_print(f"Skipped pars {hf.array2str(pars)} -- already evaluated", log)
                if np.max(np.abs(pars - starting_dot)) < 1e-5:
                    skip = False
                continue
            if hf.bad_pars(pars=pars, bounds_only=True, model=model1):
                hf.log_print(f"Skipped bad {model1} pars: {pars}", log)
                continue

            try:
                call_prices, put_prices = pricer1.price(pars=pars)
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

                f.write(f"Pars {dh.array2str(pars)} with metric {metric} = {res.fun}: {dh.array2str(res.x)}\n")
                f.flush()
                hf.log_print(f"Metric={res.fun} with {model2} params: {res.x}", log)
                hf.log_print(f"Time spent: {timedelta(seconds=(time() - t0))}\n", log)
            except ValueError as e:
                hf.log_print(f'Skipped because {e}', log)

        try:
            open(hf.get_flagfile_name(pricing_model=model1, tuning_model=model2, metric=metric), 'x').close()
        except:
            hf.log_print(f'Failed to create .flag file for {model1}, {model2}', log)
        hf.log_print("Done\n", log)


def calibrate(args: tuple):
    assert len(args) == 5
    model = args[0]
    metric = args[1]
    info = args[2]
    is_call = args[3]
    kwargs = args[4]
    start = hf.get_last_day(hf.get_filename(model=model, metric=metric, is_call=is_call))
    with open(hf.get_filename(model=model, metric=metric, is_call=is_call), 'a') as file:
        for day in range(start + 1, len(info)):
            result = optimize_model(model=model, info=info, metric=metric, day=day, local=False, **kwargs)
            file.write(f"Day {day} with func value {result.fun}: {dh.array2str(result.x)}\n")
            file.flush()
