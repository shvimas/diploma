from VG_Pricing_Integral_vectorized import price_vg
from Heston_Pricing_Integral_vectorized import price_heston
import numpy as np


models = {
    "heston": price_heston,
    "vg": price_vg
}


def model_vg(strikes: np.ndarray, pars: tuple, model: str) -> np.ndarray:
