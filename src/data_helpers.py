from math import exp
from typing import List, Tuple

import numpy as np
import scipy.optimize as opt

import black_scholes as bs
import config
import gen_pricer as gp
from helper_funcs import bad_pars
from structs import Info, Data, EvalArgs


def array2str(arr: np.ndarray) -> str:
    return ", ".join(list(map(lambda x: str(x), arr)))


def remove_options(data: Data, info: List[Info],
                   rate=.008, remove_itm: bool = False) -> Tuple[Data, List[Info]]:
    inv_factor = 1 if remove_itm else -1
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


def cut_tails(data: Data, info, min_perc=.01, max_perc=.5, min_price=10) -> Tuple[Data, List[Info]]:
    for day in range(len(info)):
        spot = info[day].spot
        cprices = data.prices[True][day]
        pprices = data.prices[False][day]
        good_calls = (cprices > min_price) & (cprices / spot > min_perc) & (cprices / spot < max_perc)
        good_puts = (pprices > min_price) & (pprices / spot > min_perc) & (pprices / spot < max_perc)
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
                                                          bounds=config.par_bounds['bs'],
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
