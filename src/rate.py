from Heston_Pricing_Integral_vectorized import price_heston
from scipy.optimize import differential_evolution
from optimization import metrics
from eval_args import EvalArgs
import numpy as np
from modeling import par_bounds


def find_opt_rate(args: EvalArgs, actual: np.ndarray) -> float:
    best_rate = 0
    best_fun = 1000
    out = open("opt_rate.txt", "w")
    for rate in [i * .0001 for i in range(1, 35)]:
        args.r = rate
        res = differential_evolution(
                func=lambda pars: metrics["RMR"](price_heston(pars, args.as_tuple()), actual),
                maxiter=50,
                bounds=par_bounds["heston"]
        )
        if best_fun > res.fun:
            best_rate = rate
        msg = "Rate: " + str(rate) + ", res.fun: " + str(res.fun) + ", best: " + str(best_rate)
        print(msg)
        out.write(msg + "\n")
    return best_rate
