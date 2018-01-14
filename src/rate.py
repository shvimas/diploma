from scipy.optimize import differential_evolution
from optimization import metrics
from eval_args import EvalArgs
import numpy as np
from modeling import par_bounds, model_prices


def find_opt_rate(args: EvalArgs, actual: np.ndarray) -> float:
    model = "ls"
    best_rate = 0
    best_fun = 1000
    out = open("params/opt_rate.txt", "w")
    for rate in [i * .0001 for i in range(1, 35)]:
        args.r = rate
        res = differential_evolution(
                func=lambda pars: metrics["RMR"](model_prices(pars=pars, args=args, model=model), actual),
                maxiter=50,
                bounds=par_bounds[model]
        )
        if best_fun > res.fun:
            best_rate = rate
        msg = f"Rate: {rate}, res.fun: {res.fun}, best: {best_rate}"
        print(msg)
        out.write(msg + "\n")
    return best_rate
