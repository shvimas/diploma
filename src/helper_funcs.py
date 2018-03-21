import csv
import re
import sys
from math import cos, sin
from typing import Tuple, List
import numpy as np
from structs import Data, Info
from config import root_dir


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


def grid(x_min: float, x_max: float, y_min: float, y_max: float, n=10) -> np.ndarray:
    x_step = (x_max - x_min) / n
    y_step = (y_max - y_min) / n
    return np.array([[x_min + i * x_step, y_min + j * y_step] for i in range(n + 1) for j in range(n + 1)])


def extract_floats(line: str) -> tuple:
    return tuple(map(lambda x: float(x), re.findall(r'[0-9.e\-+]+', line)))


def get_flagfile_name(pricing_model: str, tuning_model: str) -> str:
    return f'{root_dir}/params/.flag_{tuning_model}_with_{pricing_model}'


def log_print(msg: str, out1, out2=sys.stdout):
    print(msg, file=out1, flush=True)
    print(msg, file=out2, flush=True)


def get_starting_dot(file, from_grid: bool) -> np.ndarray:
    pattern = r'from dot (.+) with metric' if from_grid else r'Pars (.+) with metric'
    file.seek(0)
    lines = file.readlines()
    try:
        str_dot = re.search(pattern, lines[-1]).group(1)
    except AttributeError:
        raise ValueError(f'Failed to find {pattern} in {lines[-1]}')
    return np.array(extract_floats(str_dot))


def get_last_day(filename: str) -> int:
    try:
        with open(filename) as f:
            return int(re.search(r'Day (.*?)[:\s]', f.readlines()[-1]).group(1))
    except:  # FileNotFoundError or IndexError:
        return -1
