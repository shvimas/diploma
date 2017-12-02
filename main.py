from get_data import read_data
from optimization import *


def main():
    info, strikes_call, strikes_put, prices_call, prices_put = read_data("SPH2_031612.csv")

    pars_heston = np.array([0.405, 0.0098, 0.505, 0.00057, 0.04007])
    # pars_vg = (0.996575637472, -0.142224286732, 0.0954970105615)
    pars_vg = (0.999986953538, -0.173818611342, 0.0241639766491)

    day = 1
    rate = .01
    q = rate
    maturity = info[day].mat / len(info)
    spot = info[day].spot
    strikes = strikes_call[day]
    is_call = True
    args = (spot, strikes, maturity, rate, q, is_call)
    actual = prices_call[day]

    # tmp_heston = price_heston(pars_heston, args)

    tmp_vg = price_vg(pars_vg, args)

    print(tmp_vg / actual)
    # print(price_heston(pars_heston, args) / actual)

    # optimize_heston(info, strikes_call, strikes_put, prices_call, prices_put)

    optimize_vg(info, strikes_call, strikes_put, prices_call, prices_put)


if __name__ == "__main__":
    '''
    import VG_Pricing_Integral_vectorized
    a = VG_Pricing_Integral_vectorized.cf_vg(1, 2, 3, 4, 5, .1, -.1, .05)
    
    pars_heston = (5., 1., .5, .5, .2)
    pars_vg = (.1, -.1, .15)
    price_heston = price_heston(pars_heston, args_heston)
    price_vg = price_vg(pars_vg, args_vg)
    '''
    from Log_Stable_Pricing import price_ls
    pars_ls = (1.5, .05, -.05)
    args = (100, np.array([110, 120]), 1.2, .01, .01, True)
    print(price_ls(pars=pars_ls, args=args))

    main()
