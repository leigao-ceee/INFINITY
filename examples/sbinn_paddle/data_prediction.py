# code was heavily based on https://github.com/lu-group/sbinn
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lu-group/sbinn#license

import numpy as np
from scipy.integrate import odeint

data_list = np.loadtxt('Results.dat')
print('参数:')
print(data_list)

def glucose_insulin_model(
    t,
    meal_t,
    meal_q,
    Vp=3,
    Vi=11,
    Vg=10,
    E=data_list[0],
    tp=data_list[1],
    ti=data_list[2],
    td=data_list[3],
    k=data_list[4],
    Rm=data_list[5],
    a1=data_list[6],
    C1=data_list[7],
    C2=data_list[8],
    C3=100,
    C4=data_list[9],
    C5=data_list[10],
    Ub=data_list[11],
    U0=data_list[12],
    Um=data_list[13],
    Rg=data_list[14],
    alpha=data_list[15],
    beta=data_list[16],
):
    def func(y, t):
        f1 = Rm / (1 + np.exp(-y[2] / Vg / C1 + a1))
        f2 = Ub * (1 - np.exp(-y[2] / Vg / C2))
        kappa = (1 / Vi + 1 / E / ti) / C4
        f3 = (U0 + Um / (1 + (kappa * y[1]) ** (-beta))) / Vg / C3
        f4 = Rg / (1 + np.exp(alpha * (y[5] / Vp / C5 - 1)))
        IG = np.sum(
            meal_q * k * np.exp(k * (meal_t - t)) * np.heaviside(t - meal_t, 0.5)
        )
        tmp = E * (y[0] / Vp - y[1] / Vi)
        return [
            f1 - tmp - y[0] / tp,
            tmp - y[1] / ti,
            f4 + IG - f2 - f3 * y[2],
            (y[0] - y[3]) / td,
            (y[3] - y[4]) / td,
            (y[4] - y[5]) / td,
        ]

    Vp0, Vi0, Vg0 = 3, 11, 10
    y0 = [12 * Vp0, 4 * Vi0, 110 * Vg0 ** 2, 0, 0, 0]
    return odeint(func, y0, t)


meal_t = np.array([300, 650, 1100, 2000])
meal_q = np.array([60e3, 40e3, 50e3, 100e3])
t = np.arange(0, 3000, 1)[:, None]
y = glucose_insulin_model(np.ravel(t), meal_t, meal_q)

np.savetxt("glucose_pre.dat", np.hstack((t, y)))
np.savetxt("meal_pre.dat", np.hstack((meal_t, meal_q)))