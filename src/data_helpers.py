import csv
from math import exp

from structs import Info, Data
import numpy as np
from typing import List, Tuple
import re


def func(row: list, val: str) -> int:
    try:
        ans = row.index(val)
    except ValueError:
        ans = -1
    return ans


def read_data(file) -> Tuple[Data, List[Info]]:
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
    with open(file) as input, open("sorted_" + file, "w") as output:
        output.write("\n".join(sorted(list(input.readlines()), key=lambda l: l.split()[1])))


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
            if 'with func value' in line:
                yield tuple(map(lambda x: float(x), (re.search(r'.*value .*: (.*)', line).group(1).split(", "))))
            if 'Day' in line:
                continue
            raise ValueError('')
