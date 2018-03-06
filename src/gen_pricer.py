import modeling as mo
from optimization import apply_metric
from structs import EvalArgs
import numpy as np
from typing import Union, Tuple, Sequence
from scipy.optimize import OptimizeResult
from fft import FFT


class GenPricer:
    def __init__(self, model: str, market: EvalArgs, use_fft: bool):
        self.model = model
        self.market = market
        if use_fft:
            self.price_impl = lambda pars, args: FFT(model=model, args=args).price(pars=pars)
        else:
            self.price_impl = lambda pars, args: mo.models[self.model](pars=pars, args=args)

    def price(self, pars: Union[Tuple, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        assert self.market.is_call is None

        self.market.is_call = True
        call_prices = self.price_impl(pars=pars, args=self.market.as_tuple())
        self.market.is_call = False
        put_prices = self.price_impl(pars=pars, args=self.market.as_tuple())
        self.market.is_call = None

        return call_prices, put_prices

    def evaluate(self, metric: str, pars: Union[Tuple, np.ndarray], actual: np.ndarray) -> float:
        call_prices, put_prices = self.price(pars=pars)
        prices = np.append(call_prices, put_prices)
        return apply_metric(metric=metric, actual=actual, predicted=prices)

    def optimize_pars(self, metric: str, actual_calls: np.ndarray, actual_puts: np.ndarray,
                      bounds: Sequence[Tuple[float, ...]], optimizer, **kwargs) -> OptimizeResult:
        actual = np.append(actual_calls, actual_puts)
        res = optimizer(func=lambda pars: self.evaluate(pars=pars, metric=metric, actual=actual),
                        bounds=bounds,
                        **kwargs)
        return res
