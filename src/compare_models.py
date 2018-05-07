import itertools as it
import json
import re
from typing import List, Tuple

import collections as coll
import numpy as np

import data_helpers as dh
import helper_funcs
import helper_funcs as hf

metric = "RMSE"
models = ['vg', 'he', 'ls']
vg_initial_dots = helper_funcs.get_tuning_dots(pricing_model='vg', tuning_model='', metric=metric, from_grid=True)
vg_initial_dots = helper_funcs.cut_bad_pars(pars=vg_initial_dots, model='vg', bounds_only=True)


def save_scores():
    scores_per_dot = []
    for i in range(len(vg_initial_dots)):
        scores_per_dot.append([dh.array2str(vg_initial_dots[i]), coll.OrderedDict()])

    for m1, m2 in it.permutations(models, 2):
        print(m1, m2)
        with open(hf.get_tune_file_name(pricing_model=m2 if m2 != 'he' else 'heston',
                                        tuning_model=m1 if m1 != 'he' else 'heston',
                                        metric=metric)) as f:
            for line, i in zip(f.readlines(), range(len(vg_initial_dots))):
                score = hf.extract_floats(re.search(r'metric .* =(.+?):', line).group(1))[0]
                scores_per_dot[i][1][f"{m2} with {m1}"] = score

    for i in range(len(scores_per_dot)):
        scores_per_dot[i][1] = coll.OrderedDict(sorted(scores_per_dot[i][1].items()))

    with open(hf.get_scores_file_name(metric=metric), 'w') as f:
        json.dump(scores_per_dot, f, indent=4)


def parse_json(scores: list) -> List[Tuple[Tuple[float, ...], coll.OrderedDict]]:
    for i in range(len(scores)):
        scores[i] = hf.extract_floats(scores[i][0]), coll.OrderedDict(scores[i][1])
    return scores


def get_scores() -> List[Tuple[Tuple[float, ...], coll.OrderedDict]]:
    with open(hf.get_scores_file_name(metric=metric)) as f:
        return parse_json(json.load(fp=f))


def akaike(k: int, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    assert len(predicted) == len(actual)
    assert len(actual.shape) == 1
    n = len(actual)
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    return 2 * k + n * np.log(rmse)


aic = akaike


def bic(k: int, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    assert len(predicted) == len(actual)
    assert len(actual.shape) == 1
    n = len(actual)
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    return k * np.log(n) + n * np.log(rmse)
