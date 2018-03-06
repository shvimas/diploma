import ls_pricing as ls
import vg_pricing as vg
import heston_pricing as he
import numpy as np
from fft import FFT


def main():
    spot = 100
    strikes = np.array([90, 100, 110])
    t = .5
    r = .008
    q = r
    is_call = True

    args = (spot, strikes, t, r, q, is_call)
    ls_pars = (1.3, .25)
    vg_pars = (.5, -1, .1)
    he_pars = (3, .005, .001, .3, .001)

    he_prices = he.price_heston(pars=he_pars, args=args)
    vg_prices = vg.price_vg(pars=vg_pars, args=args)
    ls_prices = ls.price_ls(pars=ls_pars, args=args)

    he_prices_fft = FFT(model='heston', args=args).price(he_pars)
    ls_prices_fft = FFT(model='ls', args=args).price(ls_pars)
    vg_prices_fft = FFT(model='vg', args=args).price(vg_pars)

    for prices, prices_fft in [[he_prices, he_prices_fft], [vg_prices, vg_prices_fft], [ls_prices, ls_prices_fft]]:
        print(prices)
        print(prices_fft)
        print()
        assert np.all(np.abs(prices_fft - prices) < 1e-2)
    pass


if __name__ == '__main__':
    main()
