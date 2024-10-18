import numpy as np
import matplotlib.pyplot as plt

from helmoltz_2d_BEM.utils import apply_fit

import scienceplots
plt.style.use('science')


def modele(h, *p):
    return p[0] + p[1] * h + p[2] * (h**2) + p[3] *(h**3)

def model_label(popt, perr):
    # return fr"{popt[0]:1.1f}(\pm {perr[0]:1.1f}) + {popt[1]:1.1f}(\pm {perr[1]:.1f})h + {popt[2]:1.1f}(\pm {perr[2]:.1f})h^2 + O(h^3)"
    return fr"{popt[0]:1.1f} + {popt[1]:1.1f}h + {popt[2]:1.1f}h^2 + O(h^3)"


p0 = [10, 100, 10, 10] 


fname = "helmoltz_2d_BEM/validation/3_IE/3_IE_validation_measurement.csv"
data = np.genfromtxt(fname=fname, delimiter=',', names=True)

popt, pcov, perr = apply_fit(modele, 1/data['N_e'], data['error'], p0, names=["a_0", "a_1", "a_2", "a_3"], verbose = True)
fit_line_props = dict(marker = "o", markersize = 1.5, linestyle = "--", linewidth = 0.5)

line_props = dict(marker = "o", markersize = 4, linestyle = "--", linewidth = 1)
fit_line_props = dict(marker = "o", markersize = 1.5, linestyle = "--", linewidth = 0.5)

fig = plt.figure()
ax = fig.add_subplot()

# , c = MSColors.LIGHT_BLUE
# , c = MSColors.ORANGE
ax.plot(1/data['N_e'], data['error'], **line_props, label = "measured")
ax.plot(1/data['N_e'],  modele(1/data['N_e'], *popt), **fit_line_props, label = fr"Fit")

ax.set(xlabel = r"$h = \frac{a}{N_e}$", ylabel = r"$\frac{\left\| p - \tilde{p}^+ \right\|}{\left\| p\right\|}$")

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
ax.legend(loc='upper left')
fig.savefig(fname="Figures/IE_error.pdf")

plt.show()


