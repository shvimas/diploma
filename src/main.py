from data_helpers import read_data, array2str
from optimization import estimate_model, is_good_enough, metrics
from Heston_Pricing_Integral_vectorized import price_heston
from VG_Pricing_Integral_vectorized import price_vg
import numpy as np
import scipy.optimize as opt
from eval_args import EvalArgs


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
                    log2console=False) -> opt.OptimizeResult:

    strikes_call, strikes_put, prices_call, prices_put = \
        remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info)

    print("Optimizing Heston with " + metric + " on day " + str(day))

    with open("params/Heston_" + metric + "_good_params.txt", "a") as good:
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
                log2console=False) -> opt.OptimizeResult:

    strikes_call, strikes_put, prices_call, prices_put = \
        remove_itm_options(strikes_call, strikes_put, prices_call, prices_put, info)

    print("Optimizing VG with " + metric + " on day " + str(day))

    with open("params/VG_" + metric + "_good_params.txt", "a") as good:
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
    info, strikes_call, strikes_put, prices_call, prices_put = read_data("SPH2_031612.csv")

    # pars_heston = (5.73144671461, 0.00310912079833, 0.200295855838, 0.0131541339298, 0.0295404046434)
    pars_heston = (0.405, 0.0098, 0.505, 0.00057, 0.04007)
    # pars_vg = (0.996575637472, -0.142224286732, 0.0954970105615)
    pars_vg = (0.999728271222, -0.124716144066, 0.109217167741)

    day = 50

    market = EvalArgs(spot=info[day].spot,
                      k=strikes_call[day],
                      tau=info[day].mat,
                      r=.01,
                      q=.01,
                      call=True)

    # tune_on_near_params(model1="vg", model2="heston",
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

    def f(pars, args: tuple, prices):
        r = pars[0]
        args = EvalArgs.from_tuple(args)
        args.r = r
        args.q = r
        return metrics["RMR"](price_heston(pars=pars[1:], args=args.as_tuple()), prices)

    opt.differential_evolution(
            func=f,
            maxiter=2000,
            bounds=((.005, 2), (.00001, 6), (.00001, 1), (.00001, 1), (0, 1), (.00001, 1)),
            args=(args, actual_call)
    )

    '''
    import rate
    rate.find_opt_rate(args=EvalArgs.from_tuple(args), actual=actual_call)
    

    tmp_heston = price_heston(pars_heston, args)

    print(tmp_heston / actual_call)

    tmp_vg = price_vg(pars_vg, args)

    print(tmp_vg / actual_call)
    # print(price_heston(pars_heston, args) / actual)
    '''
    metric = "RMR"
    heston_best = open("params/best4heston_" + metric + ".txt", "w")
    vg_best = open("params/best4vg_" + metric + ".txt", "w")

    for day in range(0, len(info)):
        p1 = optimize_heston(info=info, strikes_call=strikes_call, strikes_put=strikes_put,
                             prices_call=prices_call, prices_put=prices_put,
                             metric=metric, day=day, is_call=True, log2console=log2console)
        heston_best.write("Day " + str(day) + " with " + str(p1.fun) + ": " + array2str(p1.x) + "\n")
        heston_best.flush()

        p2 = optimize_vg(info=info, strikes_call=strikes_call, strikes_put=strikes_put,
                         prices_call=prices_call, prices_put=prices_put,
                         metric=metric, day=day, is_call=True, log2console=log2console)
        vg_best.write("Day " + str(day) + " with " + str(p2.fun) + ": " + array2str(p2.x) + "\n")
        vg_best.flush()


if __name__ == "__main__":
    main()
