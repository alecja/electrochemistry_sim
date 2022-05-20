import os, sys
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.integrate as integrate
import scipy

def fermi(y, delta_G):
    return 1. / (1. + np.exp(E(y, delta_G) / kT))

def V_B(y, delta_G):
    return 0.5 * mass * omega * omega * (y_B - y) ** 2 + delta_G

def V_A(y):
    return 0.5 * mass * omega * omega * (y_A - y) ** 2

def E(y, delta_G):
    return V_B(y, delta_G) - V_A(y)

def marg_func_A(y):
    return np.exp(-(V_A(y) - V_A(y_A)) / kT)

def marg_func_B(y, delta_G):
    return np.exp(-(V_B(y, delta_G) - V_B(y_B, delta_G)) / kT)

def delta_G_func(t):
    if type(t) != type(np.array([1])):
        result_array = np.zeros((1))
        t_list = np.array([t])
    else:
        result_array = np.zeros(len(t))
        t_list = t
    for ii, t in enumerate(t_list):
        if t < tt[-1]:
            result_array[ii] = P_start - scan_rate * t + E_0_ads
        else:
            result_array[ii] = P_end + scan_rate * (t - tt[-1]) + E_0_ads
    return result_array

def delta_G_func_ads(t):
    return (P_start - scan_rate * t) * (1 - int(t / tt[-1])) + (P_end + scan_rate * (t - tt[-1])) * int(t / tt[-1])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def a(t):
    try:
        if t > tt[-1]:
            ind = np.where(np.isclose(tt, t % tt[-1]))[0][0]
            return k_func(delta_G_func(t), 'b') + k_func(delta_G_func(t), 'f')
        else:
            ind = find_nearest(tt, t)
            return k_func(delta_G_func(t), 'b') + k_func(delta_G_func(t), 'f')
    except ValueError:
        return k_func(delta_G_func(t), 'b') + k_func(delta_G_func(t), 'f')

def b(t, c_A_temp):
    try:
        if t > tt[-1]:
            ind = np.where(np.isclose(tt, t % tt[-1]))[0][0]
            return k_func(delta_G_func(t), 'f')
        else:
            ind = find_nearest(tt, t)
            return k_func(delta_G_func(t), 'f')
    except ValueError:
        return k_func(delta_G_func(t), 'f')

def int_func_A(y, *args):
    delta_G = args[0]
    return marg_func_A(y) * fermi(y, delta_G)

def int_func_B(y, *args):
    delta_G = args[0]
    return marg_func_B(y, delta_G) * (1. - fermi(y, delta_G))

def int_func_A_ds(y, *args):
    delta_G = args[0]
    return marg_func_A(y) * fermi(y, delta_G) * (1. - fermi(y, delta_G))

def int_func_B_ds(y, *args):
    delta_G = args[0]
    return marg_func_B(y, delta_G) * (1. - fermi(y, delta_G)) * fermi(y, delta_G)

def k_func(energy, dir):
    if type(energy) != type(np.array([1])):
        result_array = np.zeros((1))
        energy_list = np.array([energy])
    else:
        result_array = np.zeros(len(energy))
        energy_list = energy
    for ii, energy in enumerate(energy_list):
        if dir == 'f':
            if MH:
                if energy in k_f_dict:
                    result_array[ii] = k_f_dict[energy]
                else:
                    k_f_dict[energy] = np.longdouble(Gamma * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / norm_A)
                    result_array[ii] = k_f_dict[energy]
            else:
                result_array[ii] = Gamma * np.longdouble(np.exp(-alpha * epsilon * energy))
        if dir == 'b':
            if MH:
                if energy in k_b_dict:
                    result_array[ii] = k_b_dict[energy]
                else:
                    k_b_dict[energy] = np.longdouble(Gamma * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / norm_A) * np.exp(energy / kT)
                    result_array[ii] = k_b_dict[energy]
            else:
                result_array[ii] = Gamma * np.longdouble(np.exp((1. - alpha) * epsilon * energy))
    if len(result_array) == 1:
        return result_array[0]
    else:
        return result_array

def dk_func_ds(energy, dir):
    if dir == 'f':
        if MH:
            return epsilon * scan_rate * Gamma * integrate.quad(int_func_A_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(marg_func_A, -np.inf, np.inf)[0])
        else:
            return Gamma * np.longdouble(np.exp(-alpha * epsilon * energy)) * (scan_rate * alpha * epsilon)
    if dir == 'b':
        if MH:
            return (epsilon * scan_rate * Gamma * integrate.quad(int_func_A_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(marg_func_A, -np.inf, np.inf)[0]) + np.longdouble(Gamma * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / norm_A) * -scan_rate * epsilon) * np.exp(energy / kT)
            return -epsilon * scan_rate * Gamma * integrate.quad(int_func_B_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(marg_func_B, -np.inf, np.inf, args=(energy,))[0])
        else:
            return Gamma * np.longdouble(np.exp((1. - alpha) * epsilon * energy)) * (-scan_rate * (1. - alpha) * epsilon)

def exact_approx(u, tt):
    ds = tt[1] - tt[0]
    t = tt[-1]
    u = np.array(u)
    v = 1. - u
    return np.longdouble((np.sqrt(D * np.pi) - a(t) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1) * np.longdouble(ds * (np.sum((-a(tt[1:-1]) * u[1:] + b(tt[1:-1], v[:-1]) - b(t, v[-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0, v[0]) - b(t, v[-1])) / np.sqrt(t)) + 2. * b(t, v[-1]) * np.sqrt(t) + c_B_init * np.sqrt(D * np.pi)).flatten()[0]

MH = True
reorg = 1.0
mass = 2000
y_A = 10
y_B = -10
omega = np.sqrt(reorg / (27.211 * mass * 2 * y_A * y_A))
Gamma = 1e2
kT = 0.00095# T * 8.617e-5 / 27.211
D = 1e-5 #5.29e-8 * 5.29e-8 / 2.42e-17
epsilon = 1. / kT
alpha = 0.5
scan_rate = 1# 1.602e-19 / 2.42e-17
time_steps = 401
c_A_init = 1
c_B_init = 1 - c_A_init

P_start = 0.02
E_0_ads = 0
P_end = -0.02
dt = abs(P_start - P_end) / (time_steps * scan_rate)
ds = dt
tt = np.linspace(0, dt * time_steps, time_steps)
P_list = np.linspace(P_start, P_end, time_steps)
norm_A = integrate.quad(marg_func_A, -np.inf, np.inf)[0]
norm_B = integrate.quad(marg_func_B, -np.inf, np.inf, args=(P_start,))[0]
k_f_dict = {}
k_b_dict = {}
u_approx = np.zeros((time_steps))
u_approx[0] = c_B_init
I = np.zeros((time_steps))

for ii in range(1, time_steps):
    u_approx[ii] = exact_approx(u_approx[:ii], tt[:ii + 1])
    I[ii] = k_func(delta_G_func(tt[ii]), 'f') * (1. - u_approx[ii]) - k_func(delta_G_func(tt[ii]), 'b') * u_approx[ii]
    if ii % 1 == 0:
        print('Done with time step ', ii, 'of ', time_steps)#'iter_steps, c_B, Gamma_A, Gamma_B = ',  u_approx[ii])

u_approx_rev = np.zeros((time_steps))
u_approx_rev[0] = u_approx[-1]
I_rev = np.zeros((time_steps))
I_rev[0] = I[-1]

for ii in range(1, time_steps):
    if u_approx_rev[ii-1] > 1 or u_approx_rev[ii-1] < 0:
        sys.exit()
    u_approx_rev[ii] = exact_approx(np.concatenate((u_approx, u_approx_rev[:ii])), np.concatenate((tt, tt[-1] + tt[:ii + 1])))
    if u_approx_rev [ii] == np.nan:
        print(u_approx_rev[ii])
        sys.exit()
        u_approx_rev[ii] == 1
    if ii == time_steps - 1:
        I_rev[ii] = k_func(delta_G_func(0), 'f') * (1. - u_approx_rev[ii]) - k_func(delta_G_func(0), 'b') * u_approx_rev[ii]
    else:
        I_rev[ii] = k_func(delta_G_func(tt[-1] + tt[ii]), 'f') * (1. - u_approx_rev[ii]) - k_func(delta_G_func(tt[-1] + tt[ii]), 'b') * u_approx_rev[ii]
    if ii % 1 == 0:
        print('Done with time step ', ii, 'of ', time_steps)#'iter_steps, c_B, Gamma_A, Gamma_B = ', u_approx_rev[ii])

#pyplot.plot(P_list, 1. - u_approx, label=r'$c_{A}$'), pyplot.plot(P_list, u_approx, label=r'$c_{B}$')
#pyplot.plot(P_list[::-1], 1. - u_approx_rev, label=r'$c^{rev}_{A}$'), pyplot.plot(P_list[::-1], u_approx_rev, label=r'$c^{rev}_{B}$')
#pyplot.xlabel(r'$\Delta G$', fontsize=16), pyplot.xlim(P_start, P_end), pyplot.legend(), pyplot.show()
pyplot.plot(P_list, I, label='sol', color='b')
pyplot.xlim(P_start * 0.99, P_end * 0.99)
pyplot.plot(P_list[::-1], I_rev, label='sol_rev', color='r')
pyplot.show()
