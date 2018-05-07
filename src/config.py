root_dir = '/Users/shvimas/GitHub/diploma'
img_dir = f"{root_dir}/paper/img"

inf_price = 1e6
inf_metric = 1e6
eps = 1e-20

# for modeling
par_bounds = {
    "heston": ((1e-6, 30), (1e-10, 1), (1e-7, 4), (-1, 1), (1e-10, 1)),
    "vg":     ((1e-6, 3), (-2, 2), (1e-6, 2)),
    "ls":     ((1.00001, 1.99999), (1e-6, 2)),
    "bs":     ((1e-10, 10), )
}

named_params = {
    'heston': [['kappa', 'theta', 'sigma', 'rho', 'v0'], 5],
    'vg': [['nu', 'theta', 'sigma'], 3],
    'ls': [['alpha', 'sigma'], 2],
    'bs': [['sigma'], 1]
}