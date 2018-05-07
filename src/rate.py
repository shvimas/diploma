import re

import numpy as np

from config import par_bounds
from data_helpers import array2str
from modeling import tune_model
from structs import EvalArgs


def find_opt_rates(args: EvalArgs, actual: np.ndarray) -> dict:
    best_rates = {"heston": 0, "vg": 0, "ls": 0}
    best_fun = {"heston": 1000, "vg": 1000, "ls": 1000}

    if args.is_call:
        postfix = "call"
    else:
        postfix = "put"

    metric = "RMR"
    step = .001
    upper = 1

    with open(f"params/opt_rate_{postfix}.txt", "a+") as f:
        try:
            last = f.readlines()[-1]
            last_rate = float(re.search(r"Rate: (.+?), ", last).group(1))
            start_from = int(last_rate / step)
        except IndexError or ValueError or AttributeError:
            start_from = 1

        for rate in [i * step for i in range(start_from, int(upper / step) + 1)]:
            for model in ("heston", "vg", "ls"):
                args.r = rate
                args.q = rate
                res = tune_model(eval_args=args, bounds=par_bounds[model], model=model, local=False,
                                 metric=metric, prices=actual, polish=True, maxiter=50)

                if best_fun[model] > res.fun:
                    best_rates[model] = rate
                    best_fun[model] = res.fun
                msg = f"Rate: {rate}, model: {model}, metric: {metric}, pars: {array2str(res.x)}, " \
                      f"res.fun: {res.fun}, best: {best_rates[model]}"
                print(msg)
                f.write(msg + "\n")
                f.flush()
            f.write("\n")

        res = dict()
        vals = tuple(zip(best_rates, best_fun))
        for i in range(len(best_rates.keys())):
            res[list(best_rates.keys())[i]] = vals[i]

    return res
