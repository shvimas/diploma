import data_helpers as dh
import helper_funcs as hf
import itertools as it
import re
import json
import collections as coll
from typing import List, Tuple

models = ['vg', 'he', 'ls']
vg_initial_dots = dh.get_tuning_dots(pricing_model='vg', tuning_model='', from_grid=True)
vg_initial_dots = dh.cut_bad_pars(pars=vg_initial_dots, model='vg', bounds_only=True)


def save_scores():
    scores_per_dot = []
    for i in range(len(vg_initial_dots)):
        scores_per_dot.append((dh.array2str(vg_initial_dots[i]), coll.OrderedDict()))

    for m1, m2 in it.permutations(models, 2):
        print(m1, m2)
        with open(f"params/tune_{m2 if m2 != 'he' else 'heston'}_with_{m1 if m1 != 'he' else 'heston'}.txt") as f:
            for line, i in zip(f.readlines(), range(len(vg_initial_dots))):
                score = hf.extract_floats(re.search(r'metric MAE =(.+?):', line).group(1))[0]
                scores_per_dot[i][1][f"{m2} with {m1}"] = score

    for i in range(len(scores_per_dot)):
        scores_per_dot[i][1] = coll.OrderedDict(sorted(scores_per_dot[i][1].items()))

    with open("params/scores.json", 'w') as f:
        json.dump(scores_per_dot, f, indent=4)


def parse_json(scores: list) -> List[Tuple[Tuple[float, ...], coll.OrderedDict]]:
    for i in range(len(scores)):
        scores[i] = hf.extract_floats(scores[i][0]), coll.OrderedDict(scores[i][1])
    return scores


def get_scores() -> List[Tuple[Tuple[float, ...], coll.OrderedDict]]:
    with open(f"params/scores.json") as f:
        return parse_json(json.load(fp=f))
