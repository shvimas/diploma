import csv
from math import exp
from structs import Info, Data
import numpy as np
from typing import List, Tuple
from math import sin, cos
import re


def read_data(file) -> Tuple[Data, List[Info]]:

    def func(row: list, val: str) -> int:
        try:
            ans = row.index(val)
        except ValueError:
            ans = -1
        return ans

    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        tmp = [[str(item).replace(",", ".") for item in row] for row in reader]
        tmp = tmp[:-1]  # last row is redundant

    info = [[tmp[i][j] for j in range(0, 4)] for i in range(2, len(tmp), 2)]
    strikes = [[tmp[i][j] for j in range(5, len(tmp[i]))] for i in range(2, len(tmp), 2)]
    prices = [[tmp[i][j] for j in range(5, len(tmp[i]))] for i in range(3, len(tmp), 2)]

    puts_pos = [row.index("PUT") for row in strikes]
    end_pos = [func(row, "") for row in strikes]

    # remove trailing empty cells and split by CP
    strikes_call = [strikes[i][:puts_pos[i]] for i in range(0, len(strikes))]
    strikes_put = [strikes[i][puts_pos[i]+1:end_pos[i]] for i in range(0, len(strikes))]
    prices_call = [prices[i][:puts_pos[i]] for i in range(0, len(prices))]
    prices_put = [prices[i][puts_pos[i]+1:end_pos[i]] for i in range(0, len(prices))]

    # convert data from str to usable format
    info = list(map(lambda row: Info(*row), info))

    prices_call = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), prices_call))
    prices_put = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), prices_put))
    strikes_call = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), strikes_call))
    strikes_put = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), strikes_put))
    data = Data(scall=strikes_call, sput=strikes_put, pcall=prices_call, pput=prices_put)

    return data, info


def sort_data(file):
    with open(file) as fin, open("sorted_" + file, "w") as fout:
        fout.write("\n".join(sorted(list(fin.readlines()), key=lambda l: l.split()[1])))


def array2str(arr: np.ndarray) -> str:
    return ", ".join(list(map(lambda x: str(x), arr)))


def remove_itm_options(data: Data, info: List[Info], rate=.03) -> Tuple[Data, List[Info]]:
    for day in range(len(info)):
        spot = info[day].spot
        tau = info[day].mat / len(info)
        otm_call = data.strikes[True][day] > spot * exp(rate * tau)
        otm_put = data.strikes[False][day] < spot

        data.strikes[True][day] = data.strikes[True][day][otm_call]
        data.prices[True][day] = data.prices[True][day][otm_call]
        data.strikes[False][day] = data.strikes[False][day][otm_put]
        data.prices[False][day] = data.prices[False][day][otm_put]

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


def prepare_data(data: Data, info: List[Info]) -> Tuple[Data, List[Info]]:
    return cut_tails(data=data, info=info)


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


def rotate(a: np.ndarray, alpha: float, center: np.ndarray = None) -> np.ndarray:
    if center is None:
        center = np.mean(a, axis=0)
    rot_vec = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    return (a - center) @ rot_vec + center


def get_filename(model: str, metric: str, is_call: bool, best=True, from_dir='params') -> str:
    return f"{from_dir}/{'best' if best else 'good'}4{model}_{metric}_{'call' if is_call else 'put'}.txt"


def restore_data_from_factorized(factorized: np.ndarray, factors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return mean + factorized @ factors


def pair_max(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    if len(seq1) != len(seq2):
        raise Exception("sequences must have the same length")

    return np.array(list(map(
            lambda i: max(seq1[i], seq2[i]),
            range(len(seq1)))))


def not_less_than_zero(seq: np.ndarray) -> np.ndarray:
    try:
        len(seq)
    except TypeError:
        seq = [seq]
    return np.array(list(map(lambda i: max(seq[i], 0), range(len(seq)))))


def gen2list(g):
    a = []
    for i in g:
        a.append(i)
    return a


def grid(x_min: float, x_max: float, y_min: float, y_max: float, n=10):
    x_step = (x_max - x_min) / n
    y_step = (y_max - y_min) / n
    return [np.array((x_min + i * x_step, y_min + j * y_step)) for i in range(n + 1) for j in range(n + 1)]


def extract_floats(line: str) -> tuple:
    return tuple(map(lambda x: float(x), re.findall(r'[0-9.e\-]+', line)))


def get_pca_data(model: str, is_call: bool) -> tuple:
    with open(f'params/pca_{model}.txt', 'r') as fin:
        lines = fin.readlines()
        bounds = extract_floats(lines[0 if is_call else 3])
        factors = np.array(list(map(
            lambda arr: extract_floats(arr),
            re.findall(r'\[.+?\]', lines[1 if is_call else 4]))))
        means = np.array(extract_floats(lines[2 if is_call else 5]))
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
    raise ValueError(f"Bad model {model}")
