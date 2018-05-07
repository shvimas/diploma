from typing import Union, Tuple, Sequence

import numpy as np
from scipy.optimize import OptimizeResult

import fft
import optimization
from optimization import apply_metric
from structs import EvalArgs


class GenPricer:
    def __init__(self, model: str, market: EvalArgs, use_fft: bool, fft_alpha=None):
        self.model = model
        self.market = market
        if use_fft and model in fft.supported_models:
            self.price_impl = lambda pars, args: fft.FFT(model=model, args=args, alpha=fft_alpha).price(pars=pars)
        else:
            try:
                self.price_impl = lambda pars, args: optimization.models[self.model](pars=pars, args=args)
            except KeyError:
                raise ValueError(f"Unsupported model: {model}")

    def price(self, pars: Union[Tuple, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        assert self.market.is_call is None

        self.market.is_call = True
        call_prices = self.price_impl(pars=pars, args=self.market.as_tuple())
        self.market.is_call = False
        put_prices = self.price_impl(pars=pars, args=self.market.as_tuple())
        self.market.is_call = None

        return call_prices, put_prices

    def price_call(self, pars: Union[Tuple, np.ndarray]) -> np.ndarray:
        assert self.market.is_call is True
        return self.price_impl(pars=pars, args=self.market.as_tuple())

    def price_put(self, pars: Union[Tuple, np.ndarray]) -> np.ndarray:
        assert self.market.is_call is False
        return self.price_impl(pars=pars, args=self.market.as_tuple())

    def evaluate(self, metric: str, pars: Union[Tuple, np.ndarray], actual: np.ndarray) -> float:
        if self.market.is_call is None:
            call_prices, put_prices = self.price(pars=pars)
            prices = np.append(call_prices, put_prices)
        elif self.market.is_call:
            prices = self.price_call(pars=pars)
        else:
            prices = self.price_put(pars=pars)
        return apply_metric(metric=metric, actual=actual, predicted=prices)

    def optimize_pars(self, metric: str, actual_calls: np.ndarray, actual_puts: np.ndarray,
                      bounds: Sequence[Tuple[float, ...]], optimizer, **kwargs) -> OptimizeResult:
        if self.market.is_call is None:
            actual = np.append(actual_calls, actual_puts)
        elif self.market.is_call:
            actual = actual_calls
        else:
            actual = actual_puts
        res = optimizer(func=lambda pars: self.evaluate(pars=pars, metric=metric, actual=actual),
                        bounds=bounds,
                        **kwargs)
        return res

    @staticmethod
    def get_implied_bs_vols(market: EvalArgs, metric: str):
        raise NotImplementedError()
