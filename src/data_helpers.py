from math import exp

from helper_funcs import extract_floats
from structs import Info, Data, EvalArgs
import numpy as np
from typing import List, Tuple
import re
import black_scholes as bs
import gen_pricer as gp
import scipy.optimize as opt
import modeling as mo


def array2str(arr: np.ndarray) -> str:
    return ", ".join(list(map(lambda x: str(x), arr)))


def remove_options(data: Data, info: List[Info],
                   rate=.03, remove_itm: bool = False) -> Tuple[Data, List[Info]]:
    inv_factor = -1 if remove_itm else 1
    for day in range(len(info)):
        spot = info[day].spot
        tau = info[day].mat
        good_calls = inv_factor * data.strikes[True][day] > spot * exp(rate * tau) * inv_factor
        good_puts = inv_factor * data.strikes[False][day] < spot * exp(rate * tau) * inv_factor

        data.strikes[True][day] = data.strikes[True][day][good_calls]
        data.prices[True][day] = data.prices[True][day][good_calls]
        data.strikes[False][day] = data.strikes[False][day][good_puts]
        data.prices[False][day] = data.prices[False][day][good_puts]

    return data, info


def cut_tails(data: Data, info, min_perc=.01, min_price=10) -> Tuple[Data, List[Info]]:
    for day in range(len(info)):
        spot = info[day].spot
        good_calls = (data.prices[True][day] > min_price) & (data.prices[True][day] / spot > min_perc)
        good_puts = (data.prices[False][day] > min_price) & (data.prices[False][day] / spot > min_perc)
        data.strikes[True][day] = data.strikes[True][day][good_calls]
        data.prices[True][day] = data.prices[True][day][good_calls]
        data.strikes[False][day] = data.strikes[False][day][good_puts]
        data.prices[False][day] = data.prices[False][day][good_puts]

    return data, info


def cut_by_bs_delta(data: Data, info: List[Info], rate=0.008, disp=False) -> Tuple[Data, List[Info]]:
    for day, is_call in [(d, c) for d in range(len(info)) for c in [True, False]]:
        pricer = gp.GenPricer(model='bs',
                              market=EvalArgs.from_structure(data=data, info=info, rate=rate, day=day),
                              use_fft=False)

        result: opt.OptimizeResult = pricer.optimize_pars(metric='MAE',
                                                          actual_calls=data.prices[True][day],
                                                          actual_puts=data.prices[False][day],
                                                          bounds=mo.par_bounds['bs'],
                                                          optimizer=opt.differential_evolution,
                                                          polish=True, disp=disp)
        bs_sigma = result.x[0]
        bs_sigma = bs_sigma

        deltas = bs.bs_delta(spot=info[day].spot,
                             strikes=data.strikes[is_call][day],
                             r=rate, q=rate, t=info[day].mat,
                             bs_sigma=bs_sigma,
                             is_call=is_call)

        data.prices[is_call][day] = data.prices[is_call][day][np.abs(deltas) >= .1]
        data.strikes[is_call][day] = data.strikes[is_call][day][np.abs(deltas) >= .1]

    return data, info


def prepare_data(data: Data, info: List[Info]) -> Tuple[Data, List[Info]]:
    return cut_by_bs_delta(*cut_tails(data=data, info=info))


def extract_centers(filename: str):
    with open(filename) as f:
        for line in f.readlines():
            if 'with params' in line:
                yield tuple(map(lambda x: float(x), (re.search(r'.*with params: (.*)', line).group(1).split(", "))))
            elif 'with func value' in line:
                yield tuple(map(lambda x: float(x), (re.search(r'.*value .*: (.*)', line).group(1).split(", "))))
            elif 'Day' in line or line is '\n':
                continue
            else:
                raise ValueError(f'bad line: {line}')


try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import matplotlib.pyplot as pl


    def plot_dots(a: np.ndarray, b: np.ndarray = None, style1: str = 'ro', style2: str = 'bo', dim=2) -> None:
        assert a.shape[1] == dim
        if b is not None:
            assert b.shape[1] == dim
        if dim == 2:
            if b is not None:
                pl.plot(a[:, 0], a[:, 1], style1, b[:, 0], b[:, 1], style2)
            else:
                pl.plot(a[:, 0], a[:, 1], style1)
        elif dim == 1:
            if b is not None:
                pl.plot(range(len(a)), a, style1, range(len(b)), b, style2)
            else:
                pl.plot(range(len(a)), a, style1)
        else:
            raise ValueError('Only support dim == 1 or 2')


    def plot_and_color_2d_params(model: str, dots: np.ndarray, factors: np.ndarray, means: np.ndarray,
                                 style_bad='ro', style_good='bo', bounds_only=True) -> None:
        restored = dots @ factors + means
        is_bad = np.array(list(map(
            lambda x: bad_pars(x, bounds_only=bounds_only, model=model),
            restored
        )))
        good = dots[np.logical_not(is_bad)]
        bad = dots[is_bad]
        plot_dots(a=good, b=bad, style1=style_good, style2=style_bad)

except ImportError:
    pass


def get_pca_data(model: str) -> tuple:
    with open(f'params/pca_{model}.txt', 'r') as fin:
        lines = fin.readlines()
        bounds = extract_floats(lines[0])
        factors = np.array(list(map(
            lambda arr: extract_floats(arr),
            re.findall(r'\[.+?\]', lines[1]))))
        means = np.array(extract_floats(lines[2]))
        return bounds, factors, means


def bad_pars(pars: tuple, bounds_only: bool, model: str) -> bool:
    import ls_pricing as ls
    import vg_pricing as vg
    import heston_pricing as he
    if model == 'ls':
        return ls.bad_pars(*pars, bounds_only=bounds_only)
    elif model == 'vg':
        return vg.bad_pars(*pars, bounds_only=bounds_only)
    elif model == 'heston':
        return he.bad_pars(*pars, bounds_only=bounds_only)
    raise ValueError(f"Unknown model {model}")
