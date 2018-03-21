from structs import EvalArgs
import numpy as np
import helper_funcs as hf
from multiprocessing import Pool
from tuning import tune_model


def tune_all_models(market: EvalArgs, metric: str):
    def get_logfile(m1: str, m2: str):
        return f'log_tune_{m2}_with_{m1}.log'

    # Step 1: tune LS and Heston on VG grid
    all_args = [['vg', 'heston', '', market, metric, get_logfile('vg', 'heston'), True],
                ['vg', 'ls', '', market, metric, get_logfile('vg', 'ls'), True]]
    Pool().map(tune_model, all_args)

    # Step 2: tune LS and Heston on previously tuned(with VG) Heston and LS params
    all_args = [['heston', 'ls', 'vg', market, metric, get_logfile('heston', 'ls'), False],
                ['ls', 'heston', 'vg', market, metric, get_logfile('ls', 'heston'), False]]
    Pool().map(tune_model, all_args)

    # Step 3: tune VG on previously tuned Heston and LS params
    all_args = [['heston', 'vg', 'ls', market, metric, get_logfile('heston', 'vg'), False],
                ['ls', 'vg', 'heston', market, metric, get_logfile('ls', 'vg'), False]]
    Pool().map(tune_model, all_args)


def main() -> None:

    # need to somehow work around with overflows
    np.seterr(all='warn')

    data, info = hf.read_data("SPH2_031612.csv")

    day = 0
    metric = "MAE"
    market = EvalArgs(spot=info[day].spot,
                      k_call=data.strikes[True][day],
                      k_put=data.strikes[False][day],
                      tau=info[day].mat, r=.008, q=.008, call=None)

    tune_all_models(market=market, metric=metric)


if __name__ == "__main__":
    main()
