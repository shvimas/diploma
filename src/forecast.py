import re
from typing import List

import numpy as np

import helper_funcs as hf
import optimization
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


def forecast(model: str, pars: np.ndarray, days: range, data: Data, info: List[Info], rate=.008) -> List[np.ndarray]:
    prices = []
    for d in days:
        pricer = GenPricer(model=model,
                           market=EvalArgs.from_structure(data=data, info=info, rate=rate, day=d),
                           use_fft=True)
        prices.append(np.append(*pricer.price(pars=pars)))
    return prices


def evaluate(predicted: List[np.ndarray], actual: List[np.ndarray], metric: str, **kwargs) -> List[float]:
    if len(predicted) != len(actual):
        raise ValueError('predicted and actual are supposed to have same length')
    return list(map(lambda x: optimization.metrics[metric](*x, **kwargs), zip(predicted, actual)))
