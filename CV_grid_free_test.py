import os, sys
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.integrate as integrate
import scipy
import socket, resource, ast

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
    return P_start + scan_direction * scan_rate * t

def delta_G_func_ads(t):
    return P_start + scan_direction * scan_rate * t

def a_cb(t):
    return k_func(delta_G_func(t), 'b')

def b_cb(t, c_A_temp):
    return k_func(delta_G_func(t), 'f') * c_A_temp

def b_cb_prime(t):
    return 0

def a_ca(t):
    return k_func(delta_G_func(t), 'f')

def b_ca(t, c_B_temp):
    return k_func(delta_G_func(t), 'b') * c_B_temp

def b_ca_prime(t):
    return 0

def a_cb_ads(t, Gamma_i, Gamma_j):
    return k_func(delta_G_func(t), 'b') + k_ads_b * (Gamma_s - (Gamma_i + Gamma_j))

def b_cb_ads(t, c_A_temp, Gamma_i):
    return k_func(delta_G_func(t), 'f') * c_A_temp + k_des_b * Gamma_i

def b_cb_prime_ads(t, Gamma_i):
    return k_des_b * Gamma_i

def a_ca_ads(t, Gamma_i, Gamma_j):
    return k_func(delta_G_func(t), 'f') + k_ads_a * (Gamma_s - (Gamma_i + Gamma_j))

def b_ca_ads(t, c_B_temp, Gamma_i):
    return k_func(delta_G_func(t), 'b') * c_B_temp + k_des_a * Gamma_i

def b_ca_prime_ads(t, Gamma_i):
    return k_des_a * Gamma_i

def dGamma_a(t, Gamma_list, v):
    Gamma_a, Gamma_b = Gamma_list
    if isotherm == 'Frumkin':
        return k_ads_a * v * (Gamma_s - (Gamma_a + Gamma_b)) - k_des_a * Gamma_a * np.exp(2. * g_frum_a * Gamma_a / kT)- k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a + k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b
    if isotherm == 'Langmuir':
        return k_ads_a * v * (Gamma_s - (Gamma_a + Gamma_b)) - k_des_a * Gamma_a - k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a + k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b
    if isotherm == 'Temkin':
        return k_ads_a * v * np.exp(-(cov * char / kT) * Gamma_a) - k_des_a * np.exp((1. - cov) * char * Gamma_a / kT) - k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a + k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b

def dGamma_b(t, Gamma_list, u):
    Gamma_a, Gamma_b = Gamma_list
    if isotherm == 'Frumkin':
        return k_ads_b * u * (Gamma_s - (Gamma_a + Gamma_b)) - k_des_b * Gamma_b * np.exp(2. * g_frum_b * Gamma_b / kT) + k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a - k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b
    if isotherm == 'Langmuir':
        return k_ads_b * u * (Gamma_s - (Gamma_a + Gamma_b)) - k_des_b * Gamma_b + k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a - k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b
    if isotherm == 'Temkin':
        return k_ads_b * u * np.exp(-(cov * char / kT) * Gamma_b) - k_des_b * np.exp((1. - cov) * char * Gamma_b / kT) + k_func_ads(delta_G_func_ads(t), 'f') * Gamma_a - k_func_ads(delta_G_func_ads(t), 'b') * Gamma_b

def dGamma(t, Gamma_list, f):
    return [dGamma_a(t, Gamma_list, f[0]), dGamma_b(t, Gamma_list, f[1])]

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

def k_func_ads(energy, dir):
    if type(energy) != type(np.array([1])):
        result_array = np.zeros((1))
        energy_list = np.array([energy])
    else:
        result_array = np.zeros(len(energy))
        energy_list = energy
    for ii, energy in enumerate(energy_list):
        if dir == 'f':
            if MH:
                if energy in k_f_dict_ads:
                    result_array[ii] = k_f_dict_ads[energy]
                else:
                    k_f_dict_ads[energy] = np.longdouble(Gamma_ads * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / norm_A)
                    result_array[ii] = k_f_dict_ads[energy]
            else:
                result_array[ii] = Gamma_ads * np.longdouble(np.exp(-alpha * epsilon * energy))
        if dir == 'b':
            if MH:
                if energy in k_b_dict_ads:
                    result_array[ii] = k_b_dict_ads[energy]
                else:
                    k_b_dict_ads[energy] = np.longdouble(Gamma_ads * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / norm_A) * np.exp(energy / kT)
                    result_array[ii] = k_b_dict_ads[energy]
            else:
                result_array[ii] = Gamma_ads * np.longdouble(np.exp((1. - alpha) * epsilon * energy))
    if len(result_array) == 1:
        return result_array[0]
    else:
        return result_array

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

def exact_ans(B, A, tt):
    t = tt[-1]
    b = b_cb_prime
    a = a_cb
    u = B
    v = A
    if len(u[1:]) == 0:
        eta_b = (dt * (b_cb(0,v[0]) - b_cb_prime(dt)) / (2. * np.sqrt(dt)) + 2 * b_cb_prime(dt) * np.sqrt(dt)) / (np.sqrt(D * np.pi) + a_cb(dt) * 1.5 * np.sqrt(dt))
        chi_b = 2 * k_func(delta_G_func(dt), 'f') / (np.sqrt(D * np.pi / dt) + 1.5 * a_cb(dt))
        print('eta_b is ', eta_b, ' chi_b is ', chi_b)
    else:
        denom = np.longdouble((np.sqrt(D * np.pi) - a(t) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
        eta_b = denom * np.longdouble(ds * (np.sum((-a(tt[1:-1]) * u[1:] + (b(tt[1:-1]) + k_func(delta_G_func(tt[1:-1]), 'f') * v[1:]) - b(t)) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0) + k_func(delta_G_func(tt[0]), 'f') * v[0] - b(t) + a(tt[0]) * c_B_init) / np.sqrt(t)) + 2. * b(t) * np.sqrt(t) + c_B_init * np.sqrt(D * np.pi)).flatten()[0]
        chi_b = denom * np.longdouble(k_func(delta_G_func(t), 'f') * (ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
        print('eta_b is ', eta_b, ' chi_b is ', chi_b)
    b = b_ca_prime
    a = a_ca
    u = A
    v = B
    if len(u[1:]) == 0:
        eta_a = (dt * (b_ca(0,v[0]) - b_ca_prime(dt)) / (2. * np.sqrt(dt)) + 2 * b_ca_prime(dt) * np.sqrt(dt) + np.sqrt(D * np.pi)) / (np.sqrt(D * np.pi) + a_ca(dt) * 1.5 * np.sqrt(dt))
        chi_a = 2 * k_func(delta_G_func(dt), 'b') / (np.sqrt(D * np.pi / dt) + 1.5 * a_ca(dt))
        print('eta_a is ', eta_a, ' chi_a is ', chi_a)
    else:
        denom = np.longdouble((np.sqrt(D * np.pi) - a(t) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
        eta_a = denom * np.longdouble(ds * (np.sum((-a(tt[1:-1]) * u[1:] + (b(tt[1:-1]) + k_func(delta_G_func(tt[1:-1]), 'b') * v[1:]) - b(t)) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0) + k_func(delta_G_func(tt[0]), 'b') * v[0] - b(t) + a(tt[0]) * c_A_init) / np.sqrt(t)) + 2. * b(t) * np.sqrt(t) + c_A_init * np.sqrt(D * np.pi)).flatten()[0]
        chi_a = denom * np.longdouble(k_func(delta_G_func(t), 'b') * (ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
        print('eta_a is ', eta_a, ' chi_a is ', chi_a)
    return ((eta_a + chi_a * eta_b) / (1. - chi_a * chi_b), (eta_b + chi_b * eta_a) / (1. - chi_a * chi_b))

def exact_ans_ads(B, A, tt, Gamma_a, Gamma_b):
    t = tt[-1]
    b = b_cb_prime_ads
    a = a_cb_ads
    u = B
    v = A
    if len(u[1:]) == 0:
        eta_b = (dt * (b_cb_ads(0,v[0],Gamma_b[0]) - b_cb_prime_ads(dt,Gamma_b[1])) / (2. * np.sqrt(dt)) + 2 * b_cb_prime_ads(dt,Gamma_b[1]) * np.sqrt(dt)) / (np.sqrt(D * np.pi) - a_cb_ads(dt,Gamma_a[1],Gamma_b[1]) * 1.5 * np.sqrt(dt))
        chi_b = 2 * k_func(delta_G_func(dt), 'f') / (np.sqrt(D * np.pi / dt) - 1.5 * a_cb_ads(dt,Gamma_a[1],Gamma_b[1]))
    else:
        denom = np.longdouble((np.sqrt(D * np.pi) - a(t,Gamma_a[-1],Gamma_b[-1]) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
        eta_b = denom * np.longdouble(ds * (np.sum((-a(tt[1:-1],Gamma_a[1:-1],Gamma_b[1:-1]) * u[1:] + (b(tt[1:-1],Gamma_b[1:-1]) + k_func(delta_G_func(tt[1:-1]), 'f') * v[1:]) - b(t,Gamma_b[-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0,Gamma_b[0]) + k_func(delta_G_func(tt[0]), 'f') * v[0] - b(t,Gamma_b[-1]) + a(tt[0],Gamma_a[0],Gamma_b[0]) * c_B_init) / np.sqrt(t)) + 2. * b(t,Gamma_b[-1]) * np.sqrt(t) + c_B_init * np.sqrt(D * np.pi)).flatten()[0]
        chi_b = denom * np.longdouble(k_func(delta_G_func(t), 'f') * (ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
    b = b_ca_prime_ads
    a = a_ca_ads
    u = A
    v = B
    if len(u[1:]) == 0:
        eta_a = (dt * (b_ca_ads(0,v[0],Gamma_a[0]) - b_ca_prime_ads(dt,Gamma_a[1])) / (2. * np.sqrt(dt)) + 2 * b_ca_prime_ads(dt,Gamma_a[1]) * np.sqrt(dt) + np.sqrt(D * np.pi)) / (np.sqrt(D * np.pi) - a_ca_ads(dt,Gamma_a[1],Gamma_b[1]) * 1.5 * np.sqrt(dt))
        chi_a = 2 * k_func(delta_G_func(dt), 'b') / (np.sqrt(D * np.pi / dt) - 1.5 * a_ca_ads(dt,Gamma_a[1],Gamma_b[1]))
    else:
        denom = np.longdouble((np.sqrt(D * np.pi) - a(t,Gamma_a[-1],Gamma_b[-1]) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
        eta_a = denom * np.longdouble(ds * (np.sum((-a(tt[1:-1],Gamma_a[1:-1],Gamma_b[1:-1]) * u[1:] + (b(tt[1:-1],Gamma_a[1:-1]) + k_func(delta_G_func(tt[1:-1]), 'b') * v[1:]) - b(t,Gamma_a[1:-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0,Gamma_a[-1]) + k_func(delta_G_func(tt[0]), 'b') * v[0] - b(t,Gamma_a[-1]) + a(tt[0],Gamma_a[0],Gamma_b[0]) * c_A_init) / np.sqrt(t)) + 2. * b(t,Gamma_a[-1]) * np.sqrt(t) + c_A_init * np.sqrt(D * np.pi)).flatten()[0]
        chi_a = denom * np.longdouble(k_func(delta_G_func(t), 'b') * (ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
    return ((eta_a + chi_a * eta_b) / (1. - chi_a * chi_b), (eta_b + chi_b * eta_a) / (1. - chi_a * chi_b))

def make_jac(t, y, f):
    v, u = f
    jac_mat = np.zeros((len(y), len(y)))
    if isotherm == 'Langmuir':
        jac_mat[0,0] = -k_ads_a * v - k_des_a - k_func_ads(delta_G_func_ads(t), 'f')
        jac_mat[0,1] = k_func_ads(delta_G_func_ads(t), 'b') - k_ads_a * v
        jac_mat[1,0] = k_func_ads(delta_G_func_ads(t), 'f') - k_ads_b * u
        jac_mat[1,1] = -k_ads_b * u - k_des_b - k_func_ads(delta_G_func_ads(t), 'b')
    elif isotherm == 'Frumkin':
        jac_mat[0,0] = -k_ads_a * v - k_des_a * np.exp(2. * g_frum_a * y[0] / kT) - k_des_a * 2 * g_frum_a * y[0] * np.exp(2. * g_frum_a * y[0] / kT) / kT - k_func_ads(delta_G_func_ads(t), 'f')
        jac_mat[0,1] = k_func_ads(delta_G_func_ads(t), 'b') - k_ads_a * v
        jac_mat[1,0] = k_func_ads(delta_G_func_ads(t), 'f') - k_ads_b * u
        jac_mat[1,1] = -k_ads_b * u - k_des_b * np.exp(2. * g_frum_b * y[1] / kT) - k_des_b * 2 * g_frum_b * y[1] * np.exp(2. * g_frum_b * y[1] / kT) / kT - k_func_ads(delta_G_func_ads(t), 'b')
    else:
        print('Warning: Isotherm not yet supported')
        sys.exit()
    return jac_mat

import warnings
warnings.filterwarnings("ignore")

is_ads = True
# Which isotherm to use for adsorption
isotherm = 'Langmuir'
# Parameters for Temkin isotherm; not yet supported
cov = 0.6
char = 1e-10
# Parameters for Frumkin isotherm
# Range for each should be (-0.002 to 0.002)
g_frum_a = 0
g_frum_b = 0
# Use BV or MH ET rates; default to MH = False (Uses BV)
MH = False
# Reorganization energy parameters (in a.u.); Gamma is coupling for in-solution (non-absorbed) ET
mass = 2000
y_A = 10
y_B = -10
# User should now enter reorg instead of omega, units of reorg are eV and should have symbol $\lambda$
# Range of reorg is (0.1-2.0 eV). Should only be entered for MH
reorg = 1.0 # eV
omega = np.sqrt(reorg / (27.211 * mass * 2 * y_A * y_A))
# Gamma should now be called k_{0}^{MHC} or k_{0}^{BV}, depending on the ET theory used
# The units are s^-1, and the range is (1e-2 - 1e4 s^-1)
# Should default to zero when adsorption is turned on
Gamma = 0
# The user will enter temperature in units of Kelvin (with variable name temp), which will then be converted to kT here
# No range needed
temp = 298.
kT = 3.18e-06 * temp
# D is in units of cm^{2} / sec, and has range of (1e-6,1e-4)
D = 1e-5
epsilon = 1. / kT
# Alpha for BV ET, should be $\alpha$ in the GUI, unitless, and range (0-1). Should only be entered for BV
alpha = 0.5
#scan_rate is in units of V/sec, and user should be entering value for scan_temp. The symbol for scan_rate in the GUI should be $\nu$
#range of scan_rate is (0.0001 - 1e4) V/sec
scan_temp = 1
scan_rate = scan_temp / 27.211
# Time steps in each direction
time_steps = 400
plot_T_F = True
# How often to print 'done with time step ....'
print_step = 100

#Gamma_s is normalized saturated surface concentration, Gamma_ads is coupling for adsorbed ET
#Gamma_s range is (0 - 1), and should be shown in GUI as $\Gamma_{s}$
#Gamma_ads range is (1e-2 to 1e4), and should be shown in GUIT as $\k_{0}^{ads}$. Units of Gamma_ads are s^{-1}
Gamma_s = 1
Gamma_ads = 1e2
#k_ads range is (1e-2, 1e2), with units of s^{-1}
k_ads = 1e0
# All ads/des rates take same value by default, but can be individually changed if needed
# Each one's range should be (0.01 - 100) of k_ads
k_ads_a = k_ads
k_ads_b = k_ads
k_des_a = k_ads
k_des_b = k_ads
c_A_init = 1.
c_B_init = 1. - c_A_init
Gamma_a = np.zeros((time_steps))
if k_ads != 0:
    Gamma_a[0] = Gamma_s * (k_ads_a / k_des_a) / (1. + k_ads_a / k_des_a) * np.exp(-2 * g_frum_a / kT)
else:
    Gamma_a[0] = 0
Gamma_b = np.zeros((time_steps))
Gamma_b[0] = 0
Gamma_a_rev = np.zeros((time_steps))
Gamma_b_rev = np.zeros((time_steps))
etol = 1e-8

#P_list are the driving forces/voltages; user will enter V_start and V_end (in GUI as V_{start} and V_{end})
#No range needed
V_start = 1.0
V_end = -1.0
P_start = V_start / 27.211
P_end = V_end / 27.211
scan_direction = np.sign(P_end - P_start)
P_list = np.linspace(P_start, P_end, time_steps + 1)
dt = abs(P_start - P_end) / (time_steps * scan_rate)
ds = dt
tt = np.linspace(0, dt * time_steps, time_steps + 1)
norm_A = integrate.quad(marg_func_A, -np.inf, np.inf)[0]
norm_B = integrate.quad(marg_func_B, -np.inf, np.inf, args=(P_start,))[0]
k_f_dict = {}
k_b_dict = {}
k_f_dict_ads = {}
k_b_dict_ads = {}
u_approx = np.zeros((time_steps))
v_approx = np.zeros((time_steps))
v_approx[0] = c_A_init
u_approx[0] = c_B_init
I = np.zeros((time_steps))
I_ads = np.zeros((time_steps))
I_rev = np.zeros((time_steps))
I_ads_rev = np.zeros((time_steps))
max_steps = 20000
max_steps_uv = 3

# this will read in a parameter file in .txt format that should be structured like a python dict (see example);
# it will overwrite above parameter values
# int vs. float is important! If a value is listed above/in parameter file as int, it neeeds to be read in as int and not float
# may cause issues if this is violated
"""
try:
    file = open('CV_grid_free.txt', 'r')
    contents = file.read()
    param_dict = ast.literal_eval(contents)
    for key in param_dict:
        if type(param_dict[key]) == int or type(param_dict[key]) == bool:
            exec('%s = %d' % (key, int(param_dict[key])))
        if type(param_dict[key]) == float:
            exec('%s = %f' % (key, float(param_dict[key])))
except IOError:
    print('No external parameter file: using default parameter values from .py script')
"""
if is_ads:
    rr = integrate.ode(dGamma, jac=make_jac).set_integrator('vode', method='bdf', nsteps=20000)
    rr.set_initial_value([Gamma_a[0], Gamma_b[0]], 0.)
    for ii in range(1, time_steps):
        Gamma_a[ii] = Gamma_a[ii - 1]
        Gamma_b[ii] = Gamma_b[ii - 1]
        current_tol = 1.
        counter = 0
        temp_ans = np.array([u_approx[ii], v_approx[ii], Gamma_a[ii], Gamma_b[ii]])
        if ii > 2:
            Gamma_a[ii] = 2 * Gamma_a[ii - 1] - Gamma_a[ii - 2]
            Gamma_b[ii] = 2 * Gamma_b[ii - 1] - Gamma_b[ii - 2]
        while (np.any(np.nan_to_num(current_tol) > etol) or counter < 2) and counter < max_steps:
            rr.set_initial_value([Gamma_a[ii-1], Gamma_b[ii-1]], tt[ii-1])
            u_approx[ii] = u_approx[ii-1]
            v_approx[ii] = v_approx[ii-1]
            u_guess = u_approx[ii]
            v_guess = v_approx[ii]
            counter_uv = 0
            while counter_uv < max_steps_uv:
                u_guess = u_approx[ii]
                v_guess = v_approx[ii]
                v_approx[ii], u_approx[ii] = exact_ans_ads(u_approx[:ii], v_approx[:ii], tt[:ii+1], Gamma_a[:ii+1], Gamma_b[:ii+1])
                if u_approx[ii] < 0:
                    u_approx[ii] = 0
                if v_approx[ii] < 0:
                    v_approx[ii] = 0
                counter_uv += 1
            u_guess = u_approx[ii-1]
            v_guess = v_approx[ii-1]
            Gamma_a[ii], Gamma_b[ii] = rr.set_f_params([v_guess, u_guess]).set_jac_params([v_guess, u_guess]).integrate(rr.t + dt)
            if u_approx[ii] < 0:
                u_approx[ii] = 0
            if v_approx[ii] < 0:
                v_approx[ii] = 0
            if Gamma_a[ii] < 0:
                Gamma_a[ii] = 0
            if Gamma_b[ii] < 0:
                Gamma_b[ii] = 0
            curr_ans = np.array([u_approx[ii], v_approx[ii], Gamma_a[ii], Gamma_b[ii]])
            current_tol = np.abs(curr_ans - temp_ans) / curr_ans
            temp_ans = curr_ans
            counter += 1
        I[ii] = k_func(delta_G_func(tt[ii]), 'f') * v_approx[ii] - k_func(delta_G_func(tt[ii]), 'b') * u_approx[ii]
        I_ads[ii] = k_func_ads(delta_G_func_ads(tt[ii]), 'f') * Gamma_a[ii] - k_func_ads(delta_G_func_ads(tt[ii]), 'b') * Gamma_b[ii]
        if ii % print_step == 0:
            print('Done with time step ', ii, ' of ', time_steps)#'iter_steps, c_A, c_B, Gamma_A, Gamma_B = ', counter, v_approx[ii], u_approx[ii], Gamma_a[ii], Gamma_b[ii], v_approx[ii] + u_approx[ii])

    Gamma_a_rev[0] = Gamma_a[-1]
    Gamma_b_rev[0] = Gamma_b[-1]
    u_approx_rev = np.zeros((time_steps))
    u_approx_rev[0] = u_approx[-1]
    v_approx_rev = np.zeros((time_steps))
    v_approx_rev[0] = v_approx[-1]
    I_rev[0] = I[-1]
    I_ads_rev[0] = I_ads[-1]
    rr.set_initial_value([Gamma_a_rev[-1], Gamma_b_rev[-1]], 0)
    P_start, P_end = P_end, P_start
    scan_direction = np.sign(P_end - P_start)
    k_f_dict = {}
    k_b_dict = {}
    c_A_init = v_approx[-1]
    c_B_init = u_approx[-1]
    P_list = np.linspace(P_start, P_end, time_steps + 1)

    for ii in range(1, time_steps):
        Gamma_a_rev[ii] = Gamma_a_rev[ii - 1]
        Gamma_b_rev[ii] = Gamma_b_rev[ii - 1]
        current_tol = 1.
        counter = 0
        temp_ans = np.array([u_approx_rev[ii], v_approx_rev[ii], Gamma_a_rev[ii], Gamma_b_rev[ii]])
        if ii > 2:
            Gamma_a_rev[ii] = 2 * Gamma_a_rev[ii - 1] - Gamma_a_rev[ii - 2]
            Gamma_b_rev[ii] = 2 * Gamma_b_rev[ii - 1] - Gamma_b_rev[ii - 2]
        while (np.any(np.nan_to_num(current_tol) > etol) or counter < 2) and counter < max_steps:
            rr.set_initial_value([Gamma_a_rev[ii-1], Gamma_b_rev[ii-1]], tt[ii-1])
            u_approx_rev[ii] = u_approx_rev[ii-1]
            v_approx_rev[ii] = v_approx_rev[ii-1]
            u_guess = u_approx_rev[ii]
            v_guess = v_approx_rev[ii]
            counter_uv = 0
            while counter_uv < max_steps_uv:
                u_guess = u_approx_rev[ii]
                v_guess = v_approx_rev[ii]
                v_approx_rev[ii], u_approx_rev[ii] = exact_ans_ads(u_approx_rev[:ii], v_approx_rev[:ii], tt[:ii+1], Gamma_a_rev[:ii+1], Gamma_b_rev[:ii+1])
                if u_approx_rev[ii] < 0:
                    u_approx_rev[ii] = 0
                if v_approx_rev[ii] < 0:
                    v_approx_rev[ii] = 0
                counter_uv += 1
            u_guess = u_approx_rev[ii-1]
            v_guess = v_approx_rev[ii-1]
            Gamma_a_rev[ii], Gamma_b_rev[ii] = rr.set_f_params([v_guess, u_guess]).set_jac_params([v_guess, u_guess]).integrate(rr.t + dt)
            if u_approx_rev[ii] < 0:
                u_approx_rev[ii] = 0
            if v_approx_rev[ii] < 0:
                v_approx_rev[ii] = 0
            if Gamma_a_rev[ii] < 0:
                Gamma_a_rev[ii] = 0
            if Gamma_b_rev[ii] < 0:
                Gamma_b_rev[ii] = 0
            curr_ans = np.array([u_approx_rev[ii], v_approx_rev[ii], Gamma_a_rev[ii], Gamma_b_rev[ii]])
            current_tol = np.abs(curr_ans - temp_ans) / curr_ans
            temp_ans = curr_ans
            counter += 1
        I_rev[ii] = k_func(delta_G_func(tt[ii]), 'f') * v_approx_rev[ii] - k_func(delta_G_func(tt[ii]), 'b') * u_approx_rev[ii]
        I_ads_rev[ii] = k_func_ads(delta_G_func_ads(tt[ii]), 'f') * Gamma_a_rev[ii] - k_func_ads(delta_G_func_ads(tt[ii]), 'b') * Gamma_b_rev[ii]
        if ii % print_step == 0:
            print('Done with time step ', ii, 'of ', time_steps)#'iter_steps, c_A, c_B, Gamma_A, Gamma_B = ', counter, v_approx_rev[ii], u_approx_rev[ii], Gamma_a_rev[ii], Gamma_b_rev[ii])

else:
    Gamma_s = 0.
    Gamma_ads = 0.
    k_ads = 0.
    k_ads_a = k_ads
    k_ads_b = k_ads
    k_des_a = k_ads
    k_des_b = k_ads
    for ii in range(1, time_steps):
        v_approx[ii], u_approx[ii] = exact_ans(u_approx[:ii], v_approx[:ii], tt[:ii+1])
        if u_approx[ii] < 0:
            u_approx[ii] = 0
        if v_approx[ii] < 0:
            v_approx[ii] = 0
        u_approx_temp = u_approx[ii] / (u_approx[ii] + v_approx[ii]) * (c_A_init + c_B_init)
        v_approx[ii] = v_approx[ii] / (u_approx[ii] + v_approx[ii]) * (c_A_init + c_B_init)
        u_approx[ii] = u_approx_temp
        I[ii] = k_func(delta_G_func(tt[ii]), 'f') * v_approx[ii] - k_func(delta_G_func(tt[ii]), 'b') * u_approx[ii]
        if ii % print_step == 0:
            print('Done with time step ', ii, ' of ', time_steps)#c_A, c_B, c_A + c_B = ', v_approx[ii], u_approx[ii], v_approx[ii] + u_approx[ii])


    u_approx_rev = np.zeros((time_steps))
    u_approx_rev[0] = u_approx[-1]
    v_approx_rev = np.zeros((time_steps))
    v_approx_rev[0] = v_approx[-1]
    I_rev[0] = I[-1]
    P_start, P_end = P_end, P_start
    scan_direction = np.sign(P_end - P_start)
    k_f_dict = {}
    k_b_dict = {}
    c_A_init = v_approx[-1]
    c_B_init = u_approx[-1]
    P_list = np.linspace(P_start, P_end, time_steps + 1)

    for ii in range(1,time_steps):
        v_approx_rev[ii], u_approx_rev[ii] = exact_ans(u_approx_rev[:ii], v_approx_rev[:ii], tt[:ii + 1])
        if u_approx_rev[ii] < 0:
            u_approx_rev[ii] = 0
        if v_approx_rev[ii] < 0:
            v_approx_rev[ii] = 0
        u_approx_temp = u_approx_rev[ii] / (u_approx_rev[ii] + v_approx_rev[ii]) * (c_A_init + c_B_init)
        v_approx_rev[ii] = v_approx_rev[ii] / (u_approx_rev[ii] + v_approx_rev[ii]) * (c_A_init + c_B_init)
        u_approx_rev[ii] = u_approx_temp
        I_rev[ii] = k_func(delta_G_func(tt[ii]), 'f') * v_approx_rev[ii] - k_func(delta_G_func(tt[ii]), 'b') * u_approx_rev[ii]
        if ii % print_step == 0:
            print('Done with time step ', ii, ' of ', time_steps)#c_A, c_B, c_A + c_B = ', v_approx_rev[ii], u_approx_rev[ii], u_approx_rev[ii] + v_approx_rev[ii])

if plot_T_F:
    pyplot.plot(P_list[::-1][1:] * 27.211, I + I_ads)
    pyplot.plot(P_list[1:] * 27.211, I_rev + I_ads_rev)
    pyplot.xlabel(r'V', fontsize=24)
    pyplot.ylabel(r'I', fontsize=24)
    pyplot.tight_layout()
    pyplot.show()
    pyplot.close()

