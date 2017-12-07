from VG_Pricing_Integral_vectorized import price_vg
from Heston_Pricing_Integral_vectorized import price_heston
from Log_Stable_Pricing import price_ls
import numpy as np


models = {
    "heston": price_heston,
    "vg": price_vg,
    "ls": price_ls
}


def model_prices(pars: tuple, args: tuple, model: str) -> np.ndarray:
    return models[model](pars=pars, args=args)


# def tune_model(prices: np.ndarray, model: str, args: tuple):
