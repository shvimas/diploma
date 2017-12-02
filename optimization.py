from Heston_Pricing_Integral_vectorized import price_heston
from VG_Pricing_Integral_vectorized import price_vg
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error


def mean(seq):
    return sum(seq) / len(seq)


def mean_ratio(predicted, actual):
    tmp = list(map(
        lambda x: (x >= 1) * x + (x < 1) / x,
        predicted / actual
    ))

    return mean(tmp)


def apply_metric(predicted, actual, metric="MAE"):
    metrics = {
        "MAE": mean_absolute_error,
        "mean ratio": mean_ratio
    }

    return metrics[metric](predicted, actual)


def is_good_enough(quality: float, metric: str) -> bool:
    values = {
        "MAE": 2,
        "mean_ratio": 1.05
    }

    return quality < values[metric]


def remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info):
    for i in range(len(info)):
        spot = info[i].spot
        calls_to_remove = [strike > spot for strike in strikes_call[i]]
        puts_to_remove = [strike < spot for strike in strikes_put[i]]

        strikes_call[i] = np.array([strikes_call[i][j] for j in range(len(strikes_call[i])) if calls_to_remove[j]])
        prices_call[i] = np.array([prices_call[i][j] for j in range(len(prices_call[i])) if calls_to_remove[j]])
        strikes_put[i] = np.array([strikes_put[i][j] for j in range(len(strikes_put[i])) if puts_to_remove[j]])
        prices_put[i] = np.array(prices_put[i][j] for j in range(len(prices_put[i])) if puts_to_remove[j])

    return strikes_call, strikes_put, prices_call, prices_put


def optimize_heston(info, strikes_call, strikes_put, prices_call, prices_put):

    '''
    strikes_call, strikes_put, prices_call, prices_put = remove_itm_options(strikes_call,
                                                                            strikes_put,
                                                                            prices_call,
                                                                            prices_put,
                                                                            info)
    '''

    with open("Heston_params.txt", "w") as output:

        metric = "MAE"

        def opt_func4heston(pars, *args) -> float:
            if len(args) != 7:
                raise Exception("args should have 7 parameters: s, k, t, r, q, is_call, prices")

            k = args[1]
            prices = args[-1]
            if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
                raise Exception("strikes and prices should be np.arrays with same length")

            tmp = price_heston(pars=pars, args=args[:-1])
            quality = apply_metric(tmp, prices, metric)

            msg = metric + ": " + str(quality) + " with params: " + ", ".join(list(map(lambda x: str(x), pars)))
            print(msg)

            if is_good_enough(quality, metric):
                output.write(msg + "\n")

            return quality

        day = 0
        rate = .01
        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        is_call = True
        actual = prices_call[day]
        bounds = [(.00001, 6), (.00001, 1), (.00001, 1), (0, 1), (.00001, 1)]

        best_pars4Heston_1 = opt.differential_evolution(
            func=opt_func4heston, bounds=bounds, disp=True,
            args=(spot, strikes_call[day], maturity, rate, q, is_call, actual))

        best_pars4Heston_2 = \
            opt.basinhopping(func=opt_func4heston,
                             x0=np.array([2., 0.01, 0.47, 0.15, 0.05]),
                             niter=100,
                             minimizer_kwargs={"args": ((spot, strikes_call[day],
                                                         maturity, rate, q,
                                                         is_call, prices_call[day])),
                                               "method": "L-BFGS-B",
                                               "bounds": bounds})
    pass


def optimize_vg(info, strikes_call, strikes_put, prices_call, prices_put):


    strikes_call, strikes_put, prices_call, prices_put = remove_itm_options(strikes_call,
                                                                            strikes_put,
                                                                            prices_call,
                                                                            prices_put,
                                                                            info)


    with open("VG_params.txt", "w") as output:

        metric = "MAE"

        def opt_func4vg(pars, *args) -> float:
            if len(pars) != 3:
                raise Exception("pars should have 3 parameters: nu, theta, sigma")

            if len(args) != 7:
                raise Exception("args should have 7 parameters: s, k, tau, r, q, is_call, prices")
            k = args[1]
            prices = args[-1]

            if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
                raise Exception("strikes and prices should be np.arrays with same length")

            quality = apply_metric(price_vg(pars=pars, args=args[:-1]), prices, metric)
            msg = metric + ": " + str(quality) + " with params: " + ", ".join(list(map(lambda x: str(x), pars)))
            print(msg)

            if is_good_enough(quality, metric):
                output.write(msg + "\n")

            return quality

        day = 0
        rate = .01
        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        is_call = True
        actual = prices_call[day]
        args = (spot, strikes_call[day], maturity, rate, q, is_call, actual)
        bounds = ((1e-6, 1), (-1, 1), (1e-6, 1))

        '''
        # 1.1364, 0.3336, 0,4045
        glob_minVG = opt.brute(
            func=opt_func4vg,
            ranges=bounds,
            Ns=10,
            full_output=True,
            args=args
        )
        '''

        # 0.832095342084, -0.18049796636, 0.0318029588701
        best_pars4VG = opt.differential_evolution(
            func=opt_func4vg,
            bounds=bounds,
            args=args
        )
    pass
