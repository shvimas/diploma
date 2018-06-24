import csv
import pickle
import re
import sys
from math import cos, sin
from typing import Tuple, List

import numpy as np
import pandas as pd

import config as cfg
from structs import Data, Info


def read_data(file='SPH2_031612.csv') -> Tuple[Data, List[Info]]:
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

    info_raw = [[tmp[i][j] for j in range(0, 4)] for i in range(2, len(tmp), 2)]
    strikes_raw = [[tmp[i][j] for j in range(5, len(tmp[i]))] for i in range(2, len(tmp), 2)]
    prices_raw = [[tmp[i][j] for j in range(5, len(tmp[i]))] for i in range(3, len(tmp), 2)]

    puts_pos = [row.index("PUT") for row in strikes_raw]
    end_pos = [func(row, "") for row in strikes_raw]

    # remove trailing empty cells and split by CP
    strikes_call = [strikes_raw[i][:puts_pos[i]] for i in range(0, len(strikes_raw))]
    strikes_put = [strikes_raw[i][puts_pos[i] + 1:end_pos[i]] for i in range(0, len(strikes_raw))]
    prices_call = [prices_raw[i][:puts_pos[i]] for i in range(0, len(prices_raw))]
    prices_put = [prices_raw[i][puts_pos[i] + 1:end_pos[i]] for i in range(0, len(prices_raw))]

    # convert data from str to usable format
    info = list(map(lambda row: Info(*row), info_raw))

    prices_call = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), prices_call))
    prices_put = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), prices_put))
    strikes_call = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), strikes_call))
    strikes_put = list(map(lambda x: np.array(list(map(lambda s: float(s), x))), strikes_put))
    data = Data(scall=strikes_call, sput=strikes_put, pcall=prices_call, pput=prices_put)

    return data, info


def read_new_data(file='data2017-2018.csv') -> Tuple[Data, List[Info]]:
    info = []
    data_prices = {True: [], False: []}
    data_strikes = {True: [], False: []}
    df = pd.read_csv(file)
    df.drop(df[df.Volume <= 5].index, inplace=True)
    grouped = df.groupby(by=['Date'], sort=False)
    for inx, tmp in enumerate(grouped):
        group = tmp[1]
        data_prices[True].append(group.Price[group.Type == 'Call'].values)
        data_strikes[True].append(group.Strike[group.Type == 'Call'].values)
        data_prices[False].append(group.Price[group.Type == 'Put'].values)
        data_strikes[False].append(group.Strike[group.Type == 'Put'].values)
        date = group.Date.values[0]
        mat = group.Time.values[0]
        spot = group.Spot.values[0]
        info.append(Info(date, mat, spot, '0', mat_in_days=False))
    data = Data(data_strikes[True], data_strikes[False], data_prices[True], data_prices[False])
    return data, info


def sort_data(file):
    with open(file) as fin, open("sorted_" + file, "w") as fout:
        fout.write("\n".join(sorted(list(fin.readlines()), key=lambda l: l.split()[1])))


def array2str(arr: np.ndarray) -> str:
    return ", ".join(list(map(lambda x: str(x), arr)))


def rotate(a: np.ndarray, alpha: float, center: np.ndarray = None) -> np.ndarray:
    if center is None:
        center = np.mean(a, axis=0)
    rot_vec = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    return (a - center) @ rot_vec + center


def get_filename(model: str, metric: str, is_call: bool = None, best=True, from_dir='params') -> str:
    if is_call is None:
        strikes_postfix = 'all'
    else:
        strikes_postfix = 'call' if is_call else 'put'
    return f"{from_dir}/{'best' if best else 'good'}4{model}_{metric}_{strikes_postfix}.txt"


def restore_data_from_factorized(factorized: np.ndarray, factors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return mean + factorized @ factors


def pair_max(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    if len(seq1) != len(seq2):
        raise Exception("sequences must have the same length")

    return np.array(list(map(
        lambda i: max(seq1[i], seq2[i]),
        range(len(seq1)))))


def cut_negative(seq: np.ndarray) -> np.ndarray:
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


def grid(x_min: float, x_max: float, y_min: float, y_max: float, n=10) -> np.ndarray:
    x_step = (x_max - x_min) / n
    y_step = (y_max - y_min) / n
    return np.array([[x_min + i * x_step, y_min + j * y_step] for i in range(n + 1) for j in range(n + 1)])


def extract_floats(line: str) -> tuple:
    return tuple(map(lambda x: float(x), re.findall(r'[0-9.e\-+]+', line)))


def get_flagfile_name(pricing_model: str, tuning_model: str, metric: str) -> str:
    return f'{cfg.root_dir}/params/.flag_{tuning_model}_with_{pricing_model}_{metric}'


def get_tune_file_name(pricing_model: str, tuning_model: str, metric: str) -> str:
    return f'{cfg.root_dir}/params/tune_{tuning_model}_with_{pricing_model}_{metric}.txt'


def get_scores_file_name(metric: str) -> str:
    return f"{cfg.root_dir}/params/scores_{metric}.json"


def get_log_file_name(model: str, metric: str) -> str:
    return f"{cfg.root_dir}/params/log_calibrate_{model}_{metric}.txt"


def log_print(msg: str, out1, out2=sys.stdout):
    print(msg, file=out1, flush=True)
    print(msg, file=out2, flush=True)


def get_starting_dot(file, from_grid: bool) -> np.ndarray:
    pattern = r'Pars (.+) with metric' if from_grid else r'Pars (.+) with metric'
    file.seek(0)
    lines = file.readlines()
    try:
        str_dot = re.search(pattern, lines[-1]).group(1)
    except AttributeError:
        raise ValueError(f'Failed to find `{pattern}` in {lines[-1]}')
    return np.array(extract_floats(str_dot))


def get_last_day(filename: str) -> int:
    try:
        with open(filename) as f:
            lines = f.readlines()
            if len(lines) == 0:
                return -1
            return int(re.search(r'Day (.*?)[:\s]', lines[-1]).group(1))
    except FileNotFoundError or IndexError or AttributeError:
        return -1


def get_prepared_data(from_dir='params') -> Data:
    with open(f"{cfg.root_dir}/{from_dir}/prepared_data.pickle", 'rb') as f:
        return pickle.load(f)


def extract_centers(filename: str):
    with open(filename) as f:
        for line in f.readlines():
            if 'with params' in line:
                yield tuple(map(lambda x: float(x), (re.search(r'.*with params: (.*)', line).group(1).split(", "))))
            elif 'with func value' in line:
                yield tuple(map(lambda x: float(x), (re.search(r'.*value .*: (.*)', line).group(1).split(", "))))
            elif 'tune' in filename:
                yield tuple(map(lambda x: float(x), (re.search(r'.*: (.*)', line).group(1).split(", "))))
            elif 'Day' in line or line is '\n':
                continue
            else:
                raise ValueError(f'bad line: {line}')


def extract_floats_with_patterns(filename: str, patterns: List[str]):
    with open(filename) as f:
        for line in f.readlines():
            found = False
            for pattern in patterns:
                match = re.search(pattern, line)
                if match is not None:
                    found = True
                    yield extract_floats(match.group(1))
                    break
            if not found:
                raise ValueError(f'In {filename} bad line: {line}')


def extract_eval_values(filename: str):
    return extract_floats_with_patterns(filename=filename,
                                        patterns=[r'func value (.*):', r'metric .* = (.*):'])


def get_eval_values(model: str, metric: str, is_call: bool) -> List[float]:
    return gen2list(extract_eval_values(get_filename(model=model, metric=metric, is_call=is_call)))


def get_pca_data(model: str) -> tuple:
    with open(f'{cfg.root_dir}/params/pca_{model}.txt', 'r') as fin:
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
    raise ValueError(f"Unknown model: {model}")


def get_tuned_params(model1: str, model2: str, metric: str, drop_bad: bool, bounds_only=True) -> np.ndarray:
    pars = np.array(gen2list(extract_centers(get_tune_file_name(
        pricing_model=model1,
        tuning_model=model2,
        metric=metric))))

    if drop_bad:
        pars = pars[list(map(lambda p: not bad_pars(pars=p, bounds_only=bounds_only, model=model2), pars))]
    return pars


def get_tuning_dots(pricing_model: str, tuning_model: str, metric: str, from_grid: bool) -> np.ndarray:
    if from_grid:
        bounds, factors, means = get_pca_data(model=pricing_model)
        return grid(*bounds, n=32) @ factors + means
    else:
        # need to preserve the number of dots, so do not drop bad params
        return get_tuned_params(model1=pricing_model, model2=tuning_model, metric=metric, drop_bad=False)


def cut_bad_pars(pars: np.ndarray, model: str, bounds_only: bool) -> np.ndarray:
    return pars[~np.array(list(map(lambda par: bad_pars(pars=par, bounds_only=bounds_only, model=model), pars)))]


def get_params(filename: str, model: str) -> pd.DataFrame:
    try:
        names, num = cfg.named_params[model]
    except KeyError:
        raise ValueError(f'unknown model: {model}')
    tmp = gen2list(extract_floats_with_patterns(filename, [r'func value (.*)']))
    tmp = list(zip(*tmp))
    df = pd.DataFrame()
    df['score'] = tmp[0]
    for i in range(len(names)):
        df[names[i]] = tmp[i + 1]
    return df


def title(model: str) -> str:
    return dict(heston='Heston', he='Heston', vg='VG', ls='FMLS', bs='BS')[model]
