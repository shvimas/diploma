import csv
from info_struct import Info
import numpy as np


def func(row: list, val: str) -> int:
    try:
        ans = row.index(val)
    except ValueError:
        ans = -1
    return ans


def read_data(file):
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

    return info, strikes_call, strikes_put, prices_call, prices_put
