from multiprocessing import Pool

import numpy as np

import data_helpers as dh
import helper_funcs as hf
from structs import EvalArgs
from tuning import tune_model


def tune_all_models(market: EvalArgs, metric: str):
    def get_logfile(m1: str, m2: str):
        return f'log_tune_{m2}_with_{m1}_{metric}.log'

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

    data, info = hf.read_new_data()

    print("Preparing data...")
    try:
        data = hf.get_prepared_data()
    except FileNotFoundError:
        data, info = dh.prepare_data(data=data, info=info)
    print("Done")

    day = 0
    metric = "RMSE"
    is_call = None
    market = EvalArgs(spot=info[day].spot,
                      k_call=data.strikes[True][day],
                      k_put=data.strikes[False][day],
                      tau=info[day].mat, r=.008, q=.008, call=is_call)

    # models = ('heston', 'vg', 'ls', 'bs')
    # kwargs = [{
    #     'data': data,
    #     'rate': .008,
    #     'disp': False,
    #     'use_fft': True,
    #     'polish': True
    # }] * len(models)
    # all_args = zip(models, [metric] * len(models), [info] * len(models), [is_call] * len(models), kwargs)
    # pool = Pool()
    # pool.map(calibrate, all_args)

    tune_all_models(market=market, metric=metric)


if __name__ == "__main__":
    main()
