'''
import matplotlib.pyplot as plt
import numpy as np


def plot_vg():
    rate = .01
    q = rate
    maturity = 1.2
    spot = 1200
    strikes = 1200
    is_call = True

    pars_vg = (2.27730323539, 0.293001925073, 0.0653070204939)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    spots = np.arange(1000, 1300, 10)
    strikes = spots
    spots_grid, strikes_grid = np.meshgrid(spots, strikes)
    prices = np.array(list(map(lambda s: price_vg(pars_vg, (s, strikes, rate, q, maturity, is_call)), spots)))

    # Plot the surface.
    surf = ax.plot_surface(spots_grid, strikes_grid, prices, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
'''