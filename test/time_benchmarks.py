from datetime import timedelta, datetime
from time import time

import numpy as np
from scipy.optimize import differential_evolution

import helper_funcs as hf
from black_scholes import price_bs
from config import par_bounds, root_dir
from fft import FFT
from gen_pricer import GenPricer
from structs import EvalArgs


def td_decorator(func, times: int, seconds: bool, log_each=False):
    def wrapper(*args, **kwargs):
        scores = []
        for i in range(times):
            t0 = time()
            _ = func(*args, **kwargs)
            if seconds:
                scores.append(time() - t0)
            else:
                scores.append(timedelta(seconds=time() - t0).microseconds)
            if log_each:
                print(f'seconds for {i + 1} iteration: {time() - t0}')
        return np.array(scores).mean()

    return wrapper


def pricing():
    times = 1000

    he_pars = (17.1602851582, 0.0592887994217, 3.69111708705, -0.788742440245, 0.0396629204273)
    vg_pars = (0.838288226409, -0.188041460262, 0.179096605713)
    ls_pars = (1.45807445595, 0.115510363099)
    bs_pars = (.2,)

    spot = 1200
    t = .76
    r = .008
    q = r
    is_call = True

    for n in [1, 10, 25, 50, 100, 200]:
        strikes = np.array([i * 50 + 500 for i in range(n)])
        args = (spot, strikes, t, r, q, is_call)
        unit = td_decorator(func=lambda p: price_bs(pars=p, args=args), times=times, seconds=False)(bs_pars)
        he_time = td_decorator(func=FFT(model='heston', args=args).price, times=times, seconds=False)(he_pars)
        vg_time = td_decorator(func=FFT(model='vg', args=args).price, times=times, seconds=False)(vg_pars)
        ls_time = td_decorator(func=FFT(model='ls', args=args).price, times=times, seconds=False)(ls_pars)
        print(f"For {n} strikes: bs: {unit} heston: {he_time}, vg: {vg_time}, ls: {ls_time}")


def tuning():
    times = 5
    metric = 'RMSE'
    optimizer = differential_evolution
    actual_puts = np.ndarray([])

    spot = 1200
    t = .76
    r = .008
    q = r
    is_call = True

    logfile = open(f'{root_dir}/params/time_benchmark_{datetime.now()}.log', 'w')

    for n in [1, 10, 25, 50, 100, 200]:
        strikes = np.array([i * 50 + 500 for i in range(n)])
        args = (spot, strikes, t, r, q, is_call)
        market = EvalArgs.from_tuple((spot, strikes, t, r, q, is_call))
        call_prices = FFT(model='vg', args=args).price((0.838288226409, -0.188041460262, 0.179096605713))

        def bs_func():
            GenPricer(model='bs', market=market, use_fft=False) \
                .optimize_pars(metric=metric, optimizer=optimizer, bounds=par_bounds['bs'],
                               actual_puts=actual_puts, actual_calls=call_prices,
                               polish=True)

        t0 = time()
        unit = td_decorator(func=bs_func, times=times, seconds=True, log_each=True)()
        hf.log_print(f'BS unit time for {n} strike(s): {unit}, real time: {(time() - t0) / times}', logfile)
        for model in ['heston', 'vg', 'ls']:
            def func():
                GenPricer(model=model, market=market, use_fft=True) \
                    .optimize_pars(metric=metric, optimizer=optimizer, bounds=par_bounds[model],
                                   actual_puts=actual_puts, actual_calls=call_prices,
                                   polish=True, disp=True, maxiter=50)
            t0 = time()
            hf.log_print(f"{model}: {td_decorator(func=func, times=times, seconds=True, log_each=True)() / unit}"
                         f" units, real time: {time() - t0}", logfile)
        hf.log_print("\n", logfile)

    logfile.close()


def main():
    pricing()
    tuning()


def sqrt():
    import math

    times = 1000000

    print(td_decorator(func=lambda: math.sqrt(1000), times=times, seconds=False)())
    print(td_decorator(func=lambda: np.sqrt(1000), times=times, seconds=False)())


if __name__ == '__main__':
    main()
    sqrt()
