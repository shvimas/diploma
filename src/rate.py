from scipy.optimize import differential_evolution
from optimization import metrics
from eval_args import EvalArgs
import numpy as np
from modeling import par_bounds, model_prices
import re
from data_helpers import array2str


def find_opt_rates(args: EvalArgs, actual: np.ndarray) -> dict:
    best_rates = {"heston": 0, "vg": 0, "ls": 0}
    best_fun = {"heston": 1000, "vg": 1000, "ls": 1000}
    if args.is_call:
        postfix = "call"
    else:
        postfix = "put"
    f = open(f"params/opt_rate_{postfix}.txt", "r+")
    metric = "RMR"
    step = .0001
    upper = 1

    try:
        last = f.readlines()[-1]
        last_rate = float(re.search(r"Rate: (.+?), ", last).group(1))
        start_from = int(last_rate / step)
    except IndexError or ValueError or AttributeError:
        start_from = 1

    for rate in [i * step for i in range(start_from, int(upper / step) + 1)]:
        for model in ("heston", "vg", "ls"):
            args.r = rate
            res = differential_evolution(
                    func=lambda pars: metrics[metric](model_prices(pars=pars, args=args, model=model), actual),
                    maxiter=50,
                    bounds=par_bounds[model]
            )
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
