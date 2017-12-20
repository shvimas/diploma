from src.data_helpers import read_data, array2str
from src.optimization import estimate_model, is_good_enough
from src.Heston_Pricing_Integral_vectorized import price_heston
from src.VG_Pricing_Integral_vectorized import price_vg
import numpy as np
import scipy.optimize as opt
from src.eval_args import EvalArgs


def remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info):
    for i in range(len(info)):
        spot = info[i].spot
        calls_to_remove = [strike > spot for strike in strikes_call[i]]
        puts_to_remove = [strike < spot for strike in strikes_put[i]]

        strikes_call[i] = np.array([strikes_call[i][j]
                                    for j in range(len(strikes_call[i])) if calls_to_remove[j]])

        prices_call[i] = np.array([prices_call[i][j]
                                   for j in range(len(prices_call[i])) if calls_to_remove[j]])

        strikes_put[i] = np.array([strikes_put[i][j]
                                   for j in range(len(strikes_put[i])) if puts_to_remove[j]])

        prices_put[i] = np.array([prices_put[i][j]
                                  for j in range(len(prices_put[i])) if puts_to_remove[j]])

    return strikes_call, strikes_put, prices_call, prices_put


def optimize_heston(info: list,
                    strikes_call: list, strikes_put: list,
                    prices_call: list, prices_put: list,
                    metric: str, day: int, is_call: bool,
                    log2console=False):
    strikes_call, strikes_put, prices_call, prices_put = \
        remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info)

    with open("../params/Heston_" + metric + "_good_params.txt", "a") as good:
        good.write("Day: " + str(day) + "\n")
        model = "heston"
        actual = prices_call[day]

        def opt_func4heston(pars, *args) -> float:
            quality = estimate_model(pars, EvalArgs.from_tuple(args), model, metric, actual)

            msg = metric + ": " + str(quality) + " with params: " + array2str(pars)
            if log2console:
                print(msg)

            if is_good_enough(quality, metric):
                good.write(msg + "\n")

            return quality

        rate = .01
        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        args_heston = (spot, strikes_call[day], maturity, rate, q, is_call)
        bounds = [(4, 6), (.00001, 1), (.00001, 1), (0, 1), (.00001, 1)]

        best_pars_heston_1 = opt.differential_evolution(
            func=opt_func4heston, bounds=bounds, disp=True,
            args=args_heston)

        '''
        best_pars_heston_2 = opt.basinhopping(
            func=opt_func4heston,
            x0=np.array([2., 0.01, 0.47, 0.15, 0.05]),
            niter=100,
            minimizer_kwargs={"args": args_heston, "method": "L-BFGS-B", "bounds": bounds})
        '''
    return best_pars_heston_1


def optimize_vg(info: list,
                strikes_call: list, strikes_put: list,
                prices_call: list, prices_put: list,
                metric: str, day: int, is_call: bool,
                log2console=False):

    strikes_call, strikes_put, prices_call, prices_put = remove_itm_options(strikes_call,
                                                                            strikes_put,
                                                                            prices_call,
                                                                            prices_put,
                                                                            info)

    with open("../params/VG_" + metric + "_good_params.txt", "a") as good:
        good.write("Day: " + str(day) + "\n")
        model = "vg"
        actual = prices_call[day]

        def opt_func4vg(pars, *args) -> float:
            quality = estimate_model(pars, EvalArgs.from_tuple(args), model, metric, actual)
            msg = metric + ": " + str(quality) + " with params: " + array2str(pars)
            if log2console:
                print(msg)

            if is_good_enough(quality, metric):
                good.write(msg + "\n")

            return quality

        rate: float = .01
        q = rate
        maturity = info[day].mat / len(info)
        spot = info[day].spot
        args_vg = (spot, strikes_call[day], maturity, rate, q, is_call)
        bounds = ((1e-6, 1), (-1, 1), (1e-6, 1))

        '''
        # 1.1364, 0.3336, 0,4045
        glob_min_vg = opt.brute(
            func=opt_func4vg,
            ranges=bounds,
            Ns=10,
            full_output=True,
            args_vg=args_vg
        )
        '''

        # 0.832095342084, -0.18049796636, 0.0318029588701
        best_pars_vg: opt.OptimizeResult = opt.differential_evolution(
            func=opt_func4vg,
            bounds=bounds,
            args=args_vg
        )

    return best_pars_vg


def main():
    info, strikes_call, strikes_put, prices_call, prices_put = read_data("../SPH2_031612.csv")

    # pars_heston = (5.73144671461, 0.00310912079833, 0.200295855838, 0.0131541339298, 0.0295404046434)
    pars_heston = (0.405, 0.0098, 0.505, 0.00057, 0.04007)
    # pars_vg = (0.996575637472, -0.142224286732, 0.0954970105615)
    pars_vg = (0.999728271222, -0.124716144066, 0.109217167741)

    day = 10

    market = EvalArgs(spot=info[day].spot,
                      k=strikes_call[day],
                      tau=info[day].mat,
                      r=1.,
                      q=1.,
                      call=True)

    #tune_on_near_params(model1="vg", model2="heston",
    #                    args=market, center=pars_vg, metric="mean ratio")

    # for logging in console
    log2console = False

    rate = .01
    q = rate
    maturity = info[day].mat / len(info)
    spot = info[day].spot
    strikes = strikes_call[day]
    is_call = True
    args = (spot, strikes, maturity, rate, q, is_call)
    actual_call = prices_call[day]

    tmp_heston = price_heston(pars_heston, args)

    print(tmp_heston / actual_call)

    tmp_vg = price_vg(pars_vg, args)

    print(tmp_vg / actual_call)
    # print(price_heston(pars_heston, args) / actual)

    for day in range(0, len(info)):
        optimize_heston(info, strikes_call, strikes_put, prices_call, prices_put,
                        "MAE", day, True, log2console)
        optimize_vg(info, strikes_call, strikes_put, prices_call, prices_put,
                    "mean ratio", day, True, log2console)


if __name__ == "__main__":
    '''
    import VG_Pricing_Integral_vectorized
    a = VG_Pricing_Integral_vectorized.cf_vg(1, 2, 3, 4, 5, .1, -.1, .05)
    
    pars_heston = (5., 1., .5, .5, .2)
    pars_vg = (.1, -.1, .15)
    price_heston = price_heston(pars_heston, args_heston)
    price_vg = price_vg(pars_vg, args_vg)
    
    from Log_Stable_Pricing import price_ls
    pars_ls = (1.5, .05, -.05)
    args = (100, np.array([110, 120]), 1.2, .01, .01, True)
    print(price_ls(pars=pars_ls, args=args))
    '''

    main()