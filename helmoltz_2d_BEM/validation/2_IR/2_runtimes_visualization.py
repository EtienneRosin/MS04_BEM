import numpy as np
import matplotlib.pyplot as plt

from helmoltz_2d_BEM.utils.graphics import MSColors

import scienceplots
plt.style.use('science')

fname = "helmoltz_2d_BEM/validation/2_IR/2_IE_validation_measurement.csv"
data = np.genfromtxt(fname=fname, delimiter=',', names=True)

line_props = dict(linestyle = "--", linewidth = 0.75)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(data['N_e'], data['G_assembly_time'], **line_props, label = r"$\mathbb{G}$ assembly time",marker = "o", markersize = 4)
ax.plot(data['N_e'], data['solve_time'], **line_props, label = r"solving time", marker = "x", markersize = 3)

ax.legend()
ax.set(xlabel = r"$N_e$", ylabel = r"$T$")
fig.savefig(fname="Figures/integral_representation_runtimes.pdf")
plt.show()