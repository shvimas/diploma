import re
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

import helper_funcs as hf
import optimization
from config import named_params
from gen_pricer import GenPricer
from structs import Data, Info, EvalArgs


def get_calibrated_pars(model: str, metric: str, day: int) -> np.ndarray:
    with open(hf.get_filename(model=model, metric=metric, is_call=None)) as f:
        lines = f.readlines()
        for line in lines:
            if f"Day {day}" in line:
                pars = hf.extract_floats(re.search(r':(.+)', line).group(1))
                return np.array(pars)
        raise ValueError(f"Failed to find day {day} in file {f.name}")


def forecast(model: str, pars: np.ndarray, use_trend: bool, metric: str,
             days: range, data: Data, info: List[Info], rate=.008) -> List[np.ndarray]:
    trend = calc_trend(model=model, days=(np.array(days) - len(days)), metric=metric) if use_trend else 0
    prices = []
    for d in days:
        pricer = GenPricer(model=model,
                           market=EvalArgs.from_structure(data=data, info=info, rate=rate, day=d),
                           use_fft=True)
        prices.append(np.append(*pricer.price(pars=pars + trend * (d - days[0]))))
    return prices


def calc_trend(model: str, days: np.ndarray, metric: str) -> np.ndarray:
    def get_pars():
        for day in days:
            yield get_calibrated_pars(model=model, metric=metric, day=day)
    try:
        params = pd.DataFrame(hf.gen2list(get_pars()), columns=named_params[model][0])
    except ValueError:
        return np.zeros(named_params[model][1])
    params['day'] = np.array(days)
    regression = sm.formula.ols(formula=f"{' + '.join(named_params[model][0])} ~ day", data=params)
    res = regression.fit()
    return res.params.loc['day', ].values


def evaluate(predicted: List[np.ndarray], actual: List[np.ndarray], metric: str, **kwargs) -> List[float]:
    if len(predicted) != len(actual):
        raise ValueError('predicted and actual are supposed to have same length')
    return list(map(lambda x: optimization.metrics[metric](*x, **kwargs), zip(predicted, actual)))
