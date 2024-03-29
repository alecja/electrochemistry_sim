import os, sys, math, time
import numpy as np
import scipy.integrate as integrate
import scipy
import ast
# import socket, resource
import argparse
import warnings

warnings.filterwarnings("ignore")

# Alternative imports for python 2.x and python 3.x respectively
IS_PYTHON2 = False
try:
    import Tkinter as tk
    import tkSimpleDialog as tkd
    from tkFileDialog import askopenfilename
    from tkMessageBox import showinfo
    import tkFont
    IS_PYTHON2 = True
except ImportError:
    import tkinter as tk
    import tkinter.simpledialog as tkd
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showinfo
    import tkinter.font as tkFont

import matplotlib
import matplotlib.pyplot as plt
import platform

# try:
#     plt.rcParams['text.usetex'] = True
# except:
#     print("Cannot load TeX")

IS_MAC = False
if platform.system() == "Darwin":
    IS_MAC = True

# matplotlib.rcParams.update({'font.size': 24})

# Default values
# Frumkin param. defaults
# Range for each should be (-0.002 to 0.002)
# FIXME: G_FRUM_A_DEF should be unitless and multiplied by KT_DEF
# i.e. user enter $g_{a}$ = 1 and G_FRUM_A_DEF should be $g_{a}$ * KT_DEF
# G_FRUM_A_DEF = G_A_TEMP * KT_DEF, will need to be instantiated after KT_DEF now
G_FRUM_A_DEF = 0
G_FRUM_B_DEF = 0
# Temkin param defaults (not yet supported)
COV_DEF = 0.6
CHAR_DEF = 1e-10

# User should now enter reorg instead of omega, units of reorg are eV and should have symbol $\lambda$
# Range of reorg is (0.1-2.0 eV). Should only be entered for MH
REORG_DEF = 1.0 # eV
# Gamma should now be called k_{0}^{MHC} or k_{0}^{BV}, depending on the ET theory used
# The units are s^-1, and the range is (1e-2 - 1e4 s^-1)
# Should default to zero when adsorption is turned on 
GAMMA_DEF = 1e2
TEMP_DEF =  298.
KT_DEF = 3.18e-6 * TEMP_DEF
# Alpha for BV ET
ALPHA_DEF = .5
#scan_rate is in units of V/sec, and user should be entering value for scan_temp. The symbol for scan_rate in the GUI should be $\nu$
#range of scan_rate is (0.0001 - 1e4) V/sec
SCAN_TEMP_DEF = 1
SCAN_RATE_DEF = SCAN_TEMP_DEF / 27.211
TIME_STEPS_DEF = 401
#Gamma_s is normalized saturated surface concentration, Gamma_ads is coupling for adsorbed ET
#Gamma_s range is (0 - 1), and should be shown in GUI as $\Gamma_{s}$
#Gamma_ads range is (1e-2 to 1e4), and should be shown in GUIT as $\k_{0}^{ads}$. Units of Gamma_ads are s^{-1}
GAMMA_S_DEF = 1.0
GAMMA_ADS_DEF = 1e2
#k_ads range is (1e-2, 1e2), with units of s^{-1}
K_ADS_DEF = 1e0
K_ADS_A_DEF = K_ADS_DEF
K_ADS_B_DEF = K_ADS_DEF
K_DES_A_DEF = K_ADS_DEF
K_DES_B_DEF = K_ADS_DEF
#P_list are the driving forces/voltages; user will enter V_start and V_end (in GUI as V_{start} and V_{end})
#No range needed
V_START_DEF = 1.0
V_END_DEF = -1.0
MH_DEF = False
PLOT_T_F_DEF = True
ISOTHERM_DEF = "Langmuir"
MASS_DEF = 2000
# Adsorption energy
E_0_ADS_DEF = 0
ADS_DEF = False

# GUI parameters
BG_SIZE = 3
PRINT_STEP_DEF = 50

def heat_eq(s, *args):
    x = args[0]
    t = args[1]
    return np.exp(-(x * x) / (4. * (t - s))) / np.sqrt(4. * np.pi * (t - s))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class CV_No_Ads:
    def __init__(self, param_dict, file=None):
        if param_dict == None:
            # this will read in a parameter file in .txt format that should be structured like a python dict (see example);
            # int vs. float is important! If a value is listed above/in parameter file as int, it neeeds to be read in as int and not float
            # may cause issues if this is violated
            if file == None:
                file = ""
            try:
                file = open(file, 'r')
                contents = file.read()
                param_dict = ast.literal_eval(contents)
                for key in param_dict:
                    if type(param_dict[key]) == int or type(param_dict[key]) == bool:
                        exec('%s = %d' % (key, int(param_dict[key])))
                    if type(param_dict[key]) == float:
                        exec('%s = %f' % (key, float(param_dict[key])))
            except IOError:
                print('No input dict and no external parameter file: Using defaults in .py script')
                param_dict = {}
        elif param_dict != None and file != None:
            print('Given file and input dict, ignoring file')
        
        self.reorg = param_dict.get('reorg', REORG_DEF)
        self.Gamma = param_dict.get('gamma', GAMMA_DEF)
        self.temp = param_dict.get('temp', TEMP_DEF)
        self.alpha = param_dict.get('alpha', ALPHA_DEF)
        self.scan_temp = param_dict.get('scan_temp', SCAN_TEMP_DEF)
        self.time_steps = param_dict.get('time_steps', TIME_STEPS_DEF)
        self.V_start = param_dict.get('v_start', V_START_DEF)
        self.V_end = param_dict.get('v_end', V_END_DEF)
        self.MH = param_dict.get('mh', MH_DEF)
        self.plot_T_F = param_dict.get('plot_T_F', PLOT_T_F_DEF)

        # Non-adjustable parameters
        self.mass = MASS_DEF
        self.y_A = 10
        self.y_B = -10
        self.c_A_init = 1
        self.c_B_init = 1 - self.c_A_init
        # D is in units of cm^{2} / sec, and has range of (1e-6,1e-4)
        self.D = 1e-5 #5.29e-8 * 5.29e-8 / 2.42e-17
        

        #scan_rate is in units of V/sec, and user should be entering value for scan_temp. The symbol for scan_rate in the GUI should be $\nu$
        #range of scan_rate is (0.0001 - 1e4) V/sec TODO
        self.scan_rate = self.scan_temp / 27.211

        # Calculate additional parameters
        self.omega = np.sqrt(self.reorg / (27.211 * self.mass * 2 * self.y_A * self.y_A))
        self.kT = 3.18e-06 * self.temp# T * 8.617e-5 / 27.211
        self.epsilon = 1. / self.kT
        self.P_start = self.V_start / 27.211
        self.P_end = self.V_end / 27.211
        self.E_0_ads = 0
        dt = abs(self.P_start - self.P_end) / (self.time_steps * self.scan_rate)
        ds = dt
        self.tt = np.linspace(0, dt * self.time_steps, self.time_steps)
        self.P_list = np.linspace(self.P_start,self. P_end, self.time_steps)
        self.norm_A = integrate.quad(self.marg_func_A, -np.inf, np.inf)[0]
        self.norm_B = integrate.quad(self.marg_func_B, -np.inf, np.inf, args=(self.P_start,))[0]
        self.k_f_dict = {}
        self.k_b_dict = {}
        self.u_approx = np.zeros((self.time_steps))
        self.u_approx[0] = self.c_B_init
        self.I = np.zeros((self.time_steps))
        self.I_rev = np.zeros((self.time_steps))
        self.print_step = PRINT_STEP_DEF

        # Plotting initialization
        if self.plot_T_F:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("V", fontsize=24)
            self.ax.set_ylabel(r"$I/FAc_0 (cm^{-1}s^{-1})$", fontsize=24)
            self.ax.set_title("Simulated CV (no adsorption)")
            self.ax.set_xlim(self.P_start * 27.211, self.P_end * 27.211)
            self.ax.set_ylim(-200, 200)
            self.fig.canvas.draw()
            plt.show(block=False)
            self.for_line, = self.ax.plot([], [])
            self.rev_line, = self.ax.plot([], [])
        else:
            self.fig = None
            self.ax = None

    def fermi(self, y, delta_G):
        return 1. / (1. + np.exp(self.E(y, delta_G) / self.kT))

    def V_B(self, y, delta_G):
        return 0.5 * self.mass * self.omega * self.omega * (self.y_B - y) ** 2 + delta_G

    def V_A(self, y):
        return 0.5 * self.mass * self.omega * self.omega * (self.y_A - y) ** 2

    def E(self, y, delta_G):
        return self.V_B(y, delta_G) - self.V_A(y)

    def marg_func_A(self, y):
        return np.exp(-(self.V_A(y) - self.V_A(self.y_A)) / self.kT)

    def marg_func_B(self, y, delta_G):
        return np.exp(-(self.V_B(y, delta_G) - self.V_B(self.y_B, delta_G)) / self.kT)

    def delta_G_func(self, t):
        if type(t) != type(np.array([1])):
            result_array = np.zeros((1))
            t_list = np.array([t])
        else:
            result_array = np.zeros(len(t))
            t_list = t
        for ii, t in enumerate(t_list):
            if t < self.tt[-1]:
                result_array[ii] = self.P_start - self.scan_rate * t
            else:
                result_array[ii] = self.P_end + self.scan_rate * (t - self.tt[-1])
        return result_array

    def delta_G_func_ads(self, t):
        return (self.P_start - self.scan_rate * t) * (1 - int(t / self.tt[-1])) + (self.P_end + self.scan_rate * (t - self.tt[-1])) * int(t / self.tt[-1])

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def a(self, t):
        try:
            if t > self.tt[-1]:
                ind = np.where(np.isclose(self.tt, t % self.tt[-1]))[0][0]
                return self.k_func(self.delta_G_func(t), 'b') + self.k_func(self.delta_G_func(t), 'f')
            else:
                ind = self.find_nearest(self.tt, t)
                return self.k_func(self.delta_G_func(t), 'b') + self.k_func(self.delta_G_func(t), 'f')
        except ValueError:
            return self.k_func(self.delta_G_func(t), 'b') + self.k_func(self.delta_G_func(t), 'f')

    def b(self, t, c_A_temp):
        try:
            if t > self.tt[-1]:
                ind = np.where(np.isclose(self.tt, t % self.tt[-1]))[0][0]
                return self.k_func(self.delta_G_func(t), 'f')
            else:
                ind = self.find_nearest(self.tt, t)
                return self.k_func(self.delta_G_func(t), 'f')
        except ValueError:
            return self.k_func(self.delta_G_func(t), 'f')

    def int_func_A(self, y, *args):
        delta_G = args[0]
        return self.marg_func_A(y) * self.fermi(y, delta_G)

    def int_func_B(self, y, *args):
        delta_G = args[0]
        return self.marg_func_B(y, delta_G) * (1. - self.fermi(y, delta_G))

    def int_func_A_ds(self, y, *args):
        delta_G = args[0]
        return self.marg_func_A(y) * self.fermi(y, delta_G) * (1. - self.fermi(y, delta_G))

    def int_func_B_ds(self, y, *args):
        delta_G = args[0]
        return self.marg_func_B(y, delta_G) * (1. - self.fermi(y, delta_G)) * self.fermi(y, delta_G)

    def k_func(self, energy, dir):
        if type(energy) != type(np.array([1])):
            result_array = np.zeros((1))
            energy_list = np.array([energy])
        else:
            result_array = np.zeros(len(energy))
            energy_list = energy
        for ii, energy in enumerate(energy_list):
            if dir == 'f':
                if self.MH:
                    if energy in self.k_f_dict:
                        result_array[ii] = self.k_f_dict[energy]
                    else:
                        self.k_f_dict[energy] = np.longdouble(self.Gamma * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A)
                        result_array[ii] = self.k_f_dict[energy]
                else:
                    result_array[ii] = self.Gamma * np.longdouble(np.exp(-self.alpha * self.epsilon * energy))
            if dir == 'b':
                if self.MH:
                    if energy in self.k_b_dict:
                        result_array[ii] = self.k_b_dict[energy]
                    else:
                        self.k_b_dict[energy] = np.longdouble(self.Gamma * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A) * np.exp(energy / self.kT)
                        result_array[ii] = self.k_b_dict[energy]
                else:
                    result_array[ii] = self.Gamma * np.longdouble(np.exp((1. - self.alpha) * self.epsilon * energy))
        if len(result_array) == 1:
            return result_array[0]
        else:
            return result_array

    def dk_func_ds(self, energy, dir):
        if dir == 'f':
            if self.MH:
                return self.epsilon * self.scan_rate * self.Gamma * integrate.quad(self.int_func_A_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(self.marg_func_A, -np.inf, np.inf)[0])
            else:
                return self.Gamma * np.longdouble(np.exp(-self.alpha * self.epsilon * energy)) * (self.scan_rate * self.alpha * self.epsilon)
        if dir == 'b':
            if self.MH:
                return (self.epsilon * self.scan_rate * self.Gamma * integrate.quad(self.int_func_A_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(self.marg_func_A, -np.inf, np.inf)[0]) + np.longdouble(self.Gamma * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A) * -self.scan_rate * self.epsilon) * np.exp(energy / self.kT)
                return -epsilon * scan_rate * Gamma * integrate.quad(self.int_func_B_ds, -np.inf, np.inf, args=(energy,))[0] / (integrate.quad(marg_func_B, -np.inf, np.inf, args=(energy,))[0])
            else:
                return self.Gamma * np.longdouble(np.exp((1. - self.alpha) * self.epsilon * energy)) * (-self.scan_rate * (1. - self.alpha) * self.epsilon)

    def exact_approx(self, u, tt):
        ds = tt[1] - tt[0]
        t = tt[-1]
        u = np.array(u)
        v = 1. - u
        return np.longdouble((np.sqrt(self.D * np.pi) - self.a(t) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1) * np.longdouble(ds * (np.sum((-self.a(tt[1:-1]) * u[1:] + self.b(tt[1:-1], v[:-1]) - self.b(t, v[-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (self.b(0, v[0]) - self.b(t, v[-1])) / np.sqrt(t)) + 2. * self.b(t, v[-1]) * np.sqrt(t) + self.c_B_init * np.sqrt(self.D * np.pi)).flatten()[0]
    
    def run(self):
        for ii in range(1, self.time_steps):
            self.u_approx[ii] = self.exact_approx(self.u_approx[:ii], self.tt[:ii + 1])
            self.I[ii] = self.k_func(self.delta_G_func(self.tt[ii]), 'f') * (1. - self.u_approx[ii]) - self.k_func(self.delta_G_func(self.tt[ii]), 'b') * self.u_approx[ii]
            if ii % self.print_step == 0:
                print('Done with time step ', ii, 'of ', self.time_steps)#'iter_steps, c_B, Gamma_A, Gamma_B = ',  u_approx[ii])
                if self.plot_T_F:
                    self.plot()

        u_approx_rev = np.zeros((self.time_steps))
        u_approx_rev[0] = self.u_approx[-1]
        self.I_rev[0] = self.I[-1]

        for ii in range(1, self.time_steps):
            if u_approx_rev[ii-1] > 1 or u_approx_rev[ii-1] < 0:
                sys.exit()
            u_approx_rev[ii] = self.exact_approx(np.concatenate((self.u_approx, u_approx_rev[:ii])), np.concatenate((self.tt, self.tt[-1] + self.tt[:ii + 1])))
            if u_approx_rev [ii] == np.nan:
                print(u_approx_rev[ii])
                sys.exit()
                u_approx_rev[ii] == 1
            if ii == self.time_steps - 1:
                self.I_rev[ii] = self.k_func(self.delta_G_func(0), 'f') * (1. - u_approx_rev[ii]) - self.k_func(self.delta_G_func(0), 'b') * u_approx_rev[ii]
            else:
                self.I_rev[ii] = self.k_func(self.delta_G_func(self.tt[-1] + self.tt[ii]), 'f') * (1. - u_approx_rev[ii]) - self.k_func(self.delta_G_func(self.tt[-1] + self.tt[ii]), 'b') * u_approx_rev[ii]
            if ii % self.print_step == 0:
                print('Done with time step ', ii, 'of ', self.time_steps)#'iter_steps, c_B, Gamma_A, Gamma_B = ', u_approx_rev[ii])
                if self.plot_T_F:
                    self.plot()
        
        if self.plot_T_F:
            # fig, ax = plt.subplots()
            plt.close()
            fig, ax = plt.subplots()
            ax.set_xlabel("V", fontsize=24)
            ax.set_ylabel(r"$I/FAc_0 (cm^{-1}s^{-1})$", fontsize=24)
            ax.set_title("Simulated CV (No adsorption)")
            ax.set_xlim(self.P_start * 27.211, self.P_end * 27.211)
            ax.plot(self.P_list * 27.211, self.I)
            ax.plot(self.P_list[::-1] * 27.211, self.I_rev)
            plt.show()
        
    
    def plot(self):
        #pyplot.figure((8,8))
        self.for_line.set_data(self.P_list * 27.211, self.I)
        ylim_max = max(self.I)*1.1
        y_lim_min = min(self.I_rev)*1.1
        self.ax.set_ylim(-max(ylim_max, abs(y_lim_min)), ylim_max)
        self.rev_line.set_data(self.P_list[::-1] * 27.211, self.I_rev)
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except:
            plt.pause(1e-2)

class CV_Simulator():
    def __init__(self, param_dict, file=None):
        if param_dict == None:
            # this will read in a parameter file in .txt format that should be structured like a python dict (see example);
            # int vs. float is important! If a value is listed above/in parameter file as int, it neeeds to be read in as int and not float
            # may cause issues if this is violated
            try:
                file = open(file, 'r')
                contents = file.read()
                param_dict = ast.literal_eval(contents)
                for key in param_dict:
                    if type(param_dict[key]) == int or type(param_dict[key]) == bool:
                        exec('%s = %d' % (key, int(param_dict[key])))
                    if type(param_dict[key]) == float:
                        exec('%s = %f' % (key, float(param_dict[key])))
            except IOError:
                print('No input dict and no external parameter file: Using defaults in .py script')
                param_dict = {}
        elif param_dict != None and file != None:
            print('Given file and input dict, ignoring file')
        
        # Assign parameters from dict
        self.isotherm = param_dict.get('isotherm', ISOTHERM_DEF)
        self.g_frum_a = param_dict.get('g_frum_a', G_FRUM_A_DEF)
        self.g_frum_b = param_dict.get('g_frum_b', G_FRUM_B_DEF)
        self.cov = param_dict.get('cov', COV_DEF)
        self.char = param_dict.get('char', CHAR_DEF)
        self.reorg = param_dict.get('reorg', REORG_DEF)
        self.gamma = param_dict.get('gamma', GAMMA_DEF)
        self.temp = param_dict.get('temp', TEMP_DEF)
        self.alpha = param_dict.get('alpha', ALPHA_DEF)
        self.scan_temp = param_dict.get('scan_temp', SCAN_TEMP_DEF)
        self.time_steps = param_dict.get('time_steps', TIME_STEPS_DEF)
        self.gamma_s = param_dict.get('gamma_s', GAMMA_S_DEF)
        self.gamma_ads = param_dict.get('gamma_ads', GAMMA_ADS_DEF)
        #self.k_ads = param_dict.get('k_ads', K_ADS_DEF)
        self.k_ads_a = param_dict.get('k_ads_a', K_ADS_A_DEF)
        self.k_ads_b = param_dict.get('k_ads_b', K_ADS_B_DEF)
        self.k_des_a = param_dict.get('k_des_a', K_DES_A_DEF)
        self.k_des_b = param_dict.get('k_des_b', K_DES_B_DEF)
        self.v_start = param_dict.get('v_start', V_START_DEF)
        self.v_end = param_dict.get('v_end', V_END_DEF)
        self.mh = param_dict.get('mh', MH_DEF)
        self.plot_T_F = param_dict.get('plot_T_F', PLOT_T_F_DEF)

        # Compute additional parameters
        self.p_start = self.v_start / 27.211
        self.p_end = self.v_end / 27.211
        self.kT = 3.18e-06 * self.temp# T * 8.617e-5 / 27.211
        self.epsilon = 1. / self.kT
        self.scan_rate = self.scan_temp / 27.211
        self.c_A_init = 1.
        self.c_B_init = 1. - self.c_A_init#0.
        self.scan_direction = 1 if self.p_end > self.p_start else -1
        self.p_list = np.linspace(self.p_start, self.p_end, self.time_steps + 1)
        self.dt = abs(self.p_start - self.p_end) / (self.time_steps * self.scan_rate)
        self.ds = self.dt 
        self.Gamma_a = np.zeros((self.time_steps))
        # TODO - is this a good alternative to k_ads
        if (self.k_ads_a + self.k_ads_b + self.k_des_a + self.k_des_b) != 0:
            self.Gamma_a[0] = self.gamma_s * (self.k_ads_a / self.k_des_a) / (1. + self.k_ads_a / self.k_des_a) * np.exp(-2 * self.g_frum_a / self.kT)
        else:
            self.Gamma_a[0] = 0
        self.Gamma_b = np.zeros((self.time_steps))
        self.Gamma_b[0] = 0
        self.Gamma_a_rev = np.zeros((self.time_steps))
        self.Gamma_b_rev = np.zeros((self.time_steps))

        # Non-adjustable parameters (Add to gui/ read in if needed down the line)
        # Reorganization energy parameters (in a.u.); Gamma is coupling for in-solution (non-absorbed) ET
        self.mass = MASS_DEF
        self.y_A = 10
        self.y_B = -10
        # D is in units of cm^{2} / sec, and has range of (1e-6,1e-4)
        self.D = 1e-5
        self.omega = np.sqrt(self.reorg / (27.211 * self.mass * 2 * self.y_A * self.y_A))

        # Initialize other values
        self.tt = np.linspace(0, self.dt * self.time_steps, self.time_steps + 1)
        self.norm_A = integrate.quad(self.marg_func_A, -np.inf, np.inf)[0]
        self.norm_B = integrate.quad(self.marg_func_B, -np.inf, np.inf, args=(self.p_start,))[0]
        self.k_f_dict = {}
        self.k_b_dict = {}
        self.k_f_dict_ads = {}
        self.k_b_dict_ads = {}
        self.u_approx = np.zeros((self.time_steps))
        self.v_approx = np.zeros((self.time_steps))
        self.v_approx[0] = self.c_A_init
        self.u_approx[0] = self.c_B_init
        self.I = np.zeros((self.time_steps))
        self.I_ads = np.zeros((self.time_steps))
        self.I_rev = np.zeros((self.time_steps))
        self.I_ads_rev = np.zeros((self.time_steps))
        self.max_steps = 20000
        self.max_steps_uv = 3
        self.etol = 1e-8
        self.print_step = PRINT_STEP_DEF

        # Plotting initialization
        if self.plot_T_F:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("V", fontsize=24)
            self.ax.set_ylabel(r"$I/FAc_0 (cm^{-1}s^{-1})$", fontsize=24)
            self.ax.set_title("Simulated CV")
            self.ax.set_xlim(self.p_end * 27.211, self.p_start * 27.211)
            self.ax.set_ylim(-200, 200)
            self.fig.canvas.draw()
            plt.show(block=False)
            self.for_line, = self.ax.plot([], [])
            self.rev_line, = self.ax.plot([], [])
        else:
            self.fig = None
            self.ax = None

    def fermi(self, y, delta_G):
        return 1. / (1. + np.exp(self.E(y, delta_G) / self.kT))

    def V_B(self, y, delta_G):
        return 0.5 * self.mass * self.omega * self.omega * (self.y_B - y) ** 2 + delta_G

    def V_A(self, y):
        return 0.5 * self.mass * self.omega * self.omega * (self.y_A - y) ** 2

    def E(self, y, delta_G):
        return self.V_B(y, delta_G) - self.V_A(y)

    def marg_func_A(self, y):
        return np.exp(-(self.V_A(y) - self.V_A(self.y_A)) / self.kT)

    def marg_func_B(self, y, delta_G):
        return np.exp(-(self.V_B(y, delta_G) - self.V_B(self.y_B, delta_G)) / self.kT)

    def delta_G_func(self, t):
        return self.p_start + self.scan_direction * self.scan_rate * t

    def delta_G_func_ads(self, t):
        return self.p_start + self.scan_direction * self.scan_rate * t

    def a_cb(self, t):
        return self.k_func(self.delta_G_func(t), 'b')

    def b_cb(self, t, c_A_temp):
        return self.k_func(self.delta_G_func(t), 'f') * c_A_temp
    
    def b_cb_prime(self, t):
        return 0
    
    def a_ca(self, t):
        return self.k_func(self.delta_G_func(t), 'f')

    def b_ca(self, t, c_B_temp):
       return self.k_func(self.delta_G_func(t), 'b') * c_B_temp

    def b_ca_prime(self, t):
        return 0
    
    def a_cb_ads(self, t, Gamma_i, Gamma_j):
        return self.k_func(self.delta_G_func(t), 'b') + self.k_ads_b * (self.gamma_s - (Gamma_i + Gamma_j))

    def b_cb_ads(self, t, c_A_temp, Gamma_i):
        return self.k_func(self.delta_G_func(t), 'f') * c_A_temp + self.k_des_b * Gamma_i

    def b_cb_prime_ads(self, t, Gamma_i):
        return self.k_des_b * Gamma_i

    def a_ca_ads(self, t, Gamma_i, Gamma_j):
        return self.k_func(self.delta_G_func(t), 'f') + self.k_ads_a * (self.gamma_s - (Gamma_i + Gamma_j))

    def b_ca_ads(self, t, c_B_temp, Gamma_i):
        return self.k_func(self.delta_G_func(t), 'b') * c_B_temp + self.k_des_a * Gamma_i

    def b_ca_prime_ads(self, t, Gamma_i):
        return self.k_des_a * Gamma_i

    def dGamma_a(self, t, Gamma_list, v):
        Gamma_a, Gamma_b = Gamma_list
        if self.isotherm == 'Frumkin':
            return self.k_ads_a * v * (self.gamma_s - (Gamma_a + Gamma_b)) - self.k_des_a * Gamma_a * np.exp(2. * self.g_frum_a * Gamma_a / self.kT)- self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a + self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b
        if self.isotherm == 'Langmuir':
            return self.k_ads_a * v * (self.gamma_s - (Gamma_a + Gamma_b)) - self.k_des_a * Gamma_a - self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a + self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b
        if self.isotherm == 'Temkin':
            return self.k_ads_a * v * np.exp(-(self.cov * self.char / self.kT) * Gamma_a) - self.k_des_a * np.exp((1. - self.cov) * self.char * Gamma_a / self.kT) - self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a + self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b

    def dGamma_b(self, t, Gamma_list, u):
        Gamma_a, Gamma_b = Gamma_list
        if self.isotherm == 'Frumkin':
            return self.k_ads_b * u * (self.gamma_s - (Gamma_a + Gamma_b)) - self.k_des_b * Gamma_b * np.exp(2. * self.g_frum_b * Gamma_b / self.kT) + self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a - self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b
        if self.isotherm == 'Langmuir':
            return self.k_ads_b * u * (self.gamma_s - (Gamma_a + Gamma_b)) - self.k_des_b * Gamma_b + self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a - self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b
        if self.isotherm == 'Temkin':
            return self.k_ads_b * u * np.exp(-(self.cov * self.char / self.kT) * Gamma_b) - self.k_des_b * np.exp((1. - self.cov) * self.char * Gamma_b / self.kT) + self.k_func_ads(self.delta_G_func_ads(t), 'f') * Gamma_a - self.k_func_ads(self.delta_G_func_ads(t), 'b') * Gamma_b

    def dGamma(self, t, Gamma_list, f):
        return [self.dGamma_a(t, Gamma_list, f[0]), self.dGamma_b(t, Gamma_list, f[1])]

    def int_func_A(self, y, *args):
        delta_G = args[0]
        return self.marg_func_A(y) * self.fermi(y, delta_G)

    def int_func_B(self, y, *args):
        delta_G = args[0]
        return self.marg_func_B(y, delta_G) * (1. - self.fermi(y, delta_G))

    def int_func_A_ds(self, y, *args):
        delta_G = args[0]
        return self.marg_func_A(y) * self.fermi(y, delta_G) * (1. - self.fermi(y, delta_G))

    def int_func_B_ds(self, y, *args):
        delta_G = args[0]
        return self.marg_func_B(y, delta_G) * (1. - self.fermi(y, delta_G)) * self.fermi(y, delta_G)

    def k_func_ads(self, energy, dir):
        if type(energy) != type(np.array([1])):
            result_array = np.zeros((1))
            energy_list = np.array([energy])
        else:
            result_array = np.zeros(len(energy))
            energy_list = energy
        for ii, energy in enumerate(energy_list):
            if dir == 'f':
                if self.mh:
                    if energy in self.k_f_dict_ads:
                        result_array[ii] = self.k_f_dict_ads[energy]
                    else:
                        self.k_f_dict_ads[energy] = np.longdouble(self.gamma_ads * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A)
                        result_array[ii] = self.k_f_dict_ads[energy]
                else:
                    result_array[ii] = self.gamma_ads * np.longdouble(np.exp(-self.alpha * self.epsilon * energy))
            if dir == 'b':
                if self.mh:
                    if energy in self.k_b_dict_ads:
                        result_array[ii] = self.k_b_dict_ads[energy]
                    else:
                        self.k_b_dict_ads[energy] = np.longdouble(self.gamma_ads * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A) * np.exp(energy / self.kT)
                        result_array[ii] = self.k_b_dict_ads[energy]
                else:
                    result_array[ii] = self.gamma_ads * np.longdouble(np.exp((1. - self.alpha) * self.epsilon * energy))
        if len(result_array) == 1:
            return result_array[0]
        else:
            return result_array

    def k_func(self, energy, dir):
        if type(energy) != type(np.array([1])):
            result_array = np.zeros((1))
            energy_list = np.array([energy])
        else:
            result_array = np.zeros(len(energy))
            energy_list = energy
        for ii, energy in enumerate(energy_list):
            if dir == 'f':
                if self.mh:
                    if energy in self.k_f_dict:
                        result_array[ii] = self.k_f_dict[energy]
                    else:
                        self.k_f_dict[energy] = np.longdouble(self.gamma * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A)
                        result_array[ii] = self.k_f_dict[energy]
                else:
                    result_array[ii] = self.gamma * np.longdouble(np.exp(-self.alpha * self.epsilon * energy))
            if dir == 'b':
                if self.mh:
                    if energy in self.k_b_dict:
                        result_array[ii] = self.k_b_dict[energy]
                    else:
                        self.k_b_dict[energy] = np.longdouble(self.gamma * integrate.quad(self.int_func_A, -np.inf, np.inf, args=(energy,))[0] / self.norm_A) * np.exp(energy / self.kT)
                        result_array[ii] = self.k_b_dict[energy]
                else:
                    result_array[ii] = self.gamma * np.longdouble(np.exp((1. - self.alpha) * self.epsilon * energy))
        if len(result_array) == 1:
            return result_array[0]
        else:
            return result_array

    def exact_ans(self, B, A, tt):
        t = tt[-1]
        b = self.b_cb_prime
        a = self.a_cb
        u = B
        v = A
        if len(u[1:]) == 0:
            eta_b = (self.dt * (self.b_cb(0,v[0]) - self.b_cb_prime(self.dt)) / (2. * np.sqrt(self.dt)) + 2 * self.b_cb_prime(self.dt) * np.sqrt(self.dt)) / (np.sqrt(self.D * np.pi) + self.a_cb(self.dt) * 1.5 * np.sqrt(self.dt))
            chi_b = 2 * self.k_func(self.delta_G_func(self.dt), 'f') / (np.sqrt(self.D * np.pi / self.dt) + 1.5 * self.a_cb(self.dt))
            print('eta_b is ', eta_b, ' chi_b is ', chi_b)
        else:
            denom = np.longdouble((np.sqrt(self.D * np.pi) - a(t) * (self.ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
            eta_b = denom * np.longdouble(self.ds * (np.sum((-a(tt[1:-1]) * u[1:] + (b(tt[1:-1]) + self.k_func(self.delta_G_func(tt[1:-1]), 'f') * v[1:]) - b(t)) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0) + self.k_func(self.delta_G_func(tt[0]), 'f') * v[0] - b(t) + a(tt[0]) * self.c_B_init) / np.sqrt(t)) + 2. * b(t) * np.sqrt(t) + self.c_B_init * np.sqrt(self.D * np.pi)).flatten()[0]
            chi_b = denom * np.longdouble(self.k_func(self.delta_G_func(t), 'f') * (self.ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
            print('eta_b is ', eta_b, ' chi_b is ', chi_b)
        b = self.b_ca_prime
        a = self.a_ca
        u = A
        v = B
        if len(u[1:]) == 0:
            eta_a = (self.dt * (self.b_ca(0,v[0]) - self.b_ca_prime(self.dt)) / (2. * np.sqrt(self.dt)) + 2 * self.b_ca_prime(self.dt) * np.sqrt(self.dt) + np.sqrt(self.D * np.pi)) / (np.sqrt(self.D * np.pi) + self.a_ca(self.dt) * 1.5 * np.sqrt(self.dt))
            chi_a = 2 * self.k_func(self.delta_G_func(self.dt), 'b') / (np.sqrt(self.D * np.pi / self.dt) + 1.5 * self.a_ca(self.dt))
            print('eta_a is ', eta_a, ' chi_a is ', chi_a)
        else:
            denom = np.longdouble((np.sqrt(self.D * np.pi) - a(t) * (self.ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
            eta_a = denom * np.longdouble(self.ds * (np.sum((-a(tt[1:-1]) * u[1:] + (b(tt[1:-1]) + self.k_func(self.delta_G_func(tt[1:-1]), 'b') * v[1:]) - b(t)) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0) + self.k_func(self.delta_G_func(tt[0]), 'b') * v[0] - b(t) + a(tt[0]) * self.c_A_init) / np.sqrt(t)) + 2. * b(t) * np.sqrt(t) + self.c_A_init * np.sqrt(self.D * np.pi)).flatten()[0]
            chi_a = denom * np.longdouble(self.k_func(self.delta_G_func(t), 'b') * (self.ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
            print('eta_a is ', eta_a, ' chi_a is ', chi_a)
        return ((eta_a + chi_a * eta_b) / (1. - chi_a * chi_b), (eta_b + chi_b * eta_a) / (1. - chi_a * chi_b))

    def exact_ans_ads(self, B, A, tt, Gamma_a, Gamma_b):
        t = tt[-1]
        b = self.b_cb_prime_ads
        a = self.a_cb_ads
        u = B
        v = A
        if len(u[1:]) == 0:
            eta_b = (self.dt * (self.b_cb_ads(0,v[0],Gamma_b[0]) - self.b_cb_prime_ads(self.dt,Gamma_b[1])) / (2. * np.sqrt(self.dt)) + 2 * self.b_cb_prime_ads(self.dt,Gamma_b[1]) * np.sqrt(self.dt)) / (np.sqrt(self.D * np.pi) - self.a_cb_ads(self.dt,Gamma_a[1],Gamma_b[1]) * 1.5 * np.sqrt(self.dt))
            chi_b = 2 * self.k_func(self.delta_G_func(self.dt), 'f') / (np.sqrt(self.D * np.pi / self.dt) - 1.5 * self.a_cb_ads(self.dt,Gamma_a[1],Gamma_b[1]))
        else:
            denom = np.longdouble((np.sqrt(self.D * np.pi) - a(t,Gamma_a[-1],Gamma_b[-1]) * (self.ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
            eta_b = denom * np.longdouble(self.ds * (np.sum((-a(tt[1:-1],Gamma_a[1:-1],Gamma_b[1:-1]) * u[1:] + (b(tt[1:-1],Gamma_b[1:-1]) + self.k_func(self.delta_G_func(tt[1:-1]), 'f') * v[1:]) - b(t,Gamma_b[-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0,Gamma_b[0]) + self.k_func(self.delta_G_func(tt[0]), 'f') * v[0] - b(t,Gamma_b[-1]) + a(tt[0],Gamma_a[0],Gamma_b[0]) * self.c_B_init) / np.sqrt(t)) + 2. * b(t,Gamma_b[-1]) * np.sqrt(t) + self.c_B_init * np.sqrt(self.D * np.pi)).flatten()[0]
            chi_b = denom * np.longdouble(self.k_func(self.delta_G_func(t), 'f') * (self.ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
        b = self.b_ca_prime_ads
        a = self.a_ca_ads
        u = A
        v = B
        if len(u[1:]) == 0:
            eta_a = (self.dt * (self.b_ca_ads(0,v[0],Gamma_a[0]) - self.b_ca_prime_ads(self.dt,Gamma_a[1])) / (2. * np.sqrt(self.dt)) + 2 * self.b_ca_prime_ads(self.dt,Gamma_a[1]) * np.sqrt(self.dt) + np.sqrt(self.D * np.pi)) / (np.sqrt(self.D * np.pi) - self.a_ca_ads(self.dt,Gamma_a[1],Gamma_b[1]) * 1.5 * np.sqrt(self.dt))
            chi_a = 2 * self.k_func(self.delta_G_func(self.dt), 'b') / (np.sqrt(self.D * np.pi / self.dt) - 1.5 * self.a_ca_ads(self.dt,Gamma_a[1],Gamma_b[1]))
        else:
            denom = np.longdouble((np.sqrt(self.D * np.pi) - a(t,Gamma_a[-1],Gamma_b[-1]) * (self.ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1)
            eta_a = denom * np.longdouble(self.ds * (np.sum((-a(tt[1:-1],Gamma_a[1:-1],Gamma_b[1:-1]) * u[1:] + (b(tt[1:-1],Gamma_a[1:-1]) + self.k_func(self.delta_G_func(tt[1:-1]), 'b') * v[1:]) - b(t,Gamma_a[1:-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0,Gamma_a[-1]) + self.k_func(self.delta_G_func(tt[0]), 'b') * v[0] - b(t,Gamma_a[-1]) + a(tt[0],Gamma_a[0],Gamma_b[0]) * self.c_A_init) / np.sqrt(t)) + 2. * b(t,Gamma_a[-1]) * np.sqrt(t) + self.c_A_init * np.sqrt(self.D * np.pi)).flatten()[0]
            chi_a = denom * np.longdouble(self.k_func(self.delta_G_func(t), 'b') * (self.ds * np.sum(-1. / np.sqrt(t - tt[1:-1]) - 0.5 / np.sqrt(t)) + 2. * np.sqrt(t)))
        return ((eta_a + chi_a * eta_b) / (1. - chi_a * chi_b), (eta_b + chi_b * eta_a) / (1. - chi_a * chi_b))

    def make_jac(self, t, y, f):
        v, u = f
        jac_mat = np.zeros((len(y), len(y)))
        if self.isotherm == 'Langmuir':
            jac_mat[0,0] = -self.k_ads_a * v - self.k_des_a - self.k_func_ads(self.delta_G_func_ads(t), 'f')
            jac_mat[0,1] = self.k_func_ads(self.delta_G_func_ads(t), 'b') - self.k_ads_a * v
            jac_mat[1,0] = self.k_func_ads(self.delta_G_func_ads(t), 'f') - self.k_ads_b * u
            jac_mat[1,1] = -self.k_ads_b * u - self.k_des_b - self.k_func_ads(self.delta_G_func_ads(t), 'b')
        elif self.isotherm == 'Frumkin':
            jac_mat[0,0] = -self.k_ads_a * v - self.k_des_a * np.exp(2. * self.g_frum_a * y[0] / self.kT) - self.k_des_a * 2 * self.g_frum_a * y[0] * np.exp(2. * self.g_frum_a * y[0] / self.kT) / self.kT - self.k_func_ads(self.delta_G_func_ads(t), 'f')
            jac_mat[0,1] = self.k_func_ads(self.delta_G_func_ads(t), 'b') - self.k_ads_a * v
            jac_mat[1,0] = self.k_func_ads(self.delta_G_func_ads(t), 'f') - self.k_ads_b * u
            jac_mat[1,1] = -self.k_ads_b * u - self.k_des_b * np.exp(2. * self.g_frum_b * y[1] / self.kT) - self.k_des_b * 2 * self.g_frum_b * y[1] * np.exp(2. * self.g_frum_b * y[1] / self.kT) / self.kT - self.k_func_ads(self.delta_G_func_ads(t), 'b')
        else:
            jac_mat[0, 0] = -self.kT/self.cov/self.char*self.k_ads_a*v*np.exp(-self.cov*self.char*y[0]/self.kT) - self.kT/(1-self.cov)/self.char*self.k_des_a*np.exp((1-self.cov)*self.char*y[0]/self.kT) - self.k_func_ads(self.delta_G_func_ads(t), 'f')
            jac_mat[0, 1] = self.k_func_ads(self.delta_G_func_ads(t), 'b')
            jac_mat[1, 0] = self.k_func_ads(self.delta_G_func_ads(t), 'f')
            jac_mat[1, 1] = -self.kT/self.cov/self.char*self.k_ads_b*u*np.exp(self.cov*self.char*y[1]/self.kT) - self.kT/(1-self.cov)/self.char*self.k_des_b*np.exp((1-self.cov)*self.char*y[1]/self.kT) - self.k_func_ads(self.delta_G_func_ads(t), 'b')
            print('Warning: Isotherm not yet supported')
            sys.exit()
        return jac_mat

    def run(self):
        rr = integrate.ode(self.dGamma, jac=self.make_jac).set_integrator('vode', method='bdf', nsteps=20000)
        rr.set_initial_value([self.Gamma_a[0], self.Gamma_b[0]], 0.)
        for ii in range(1, self.time_steps):
            self.Gamma_a[ii] = self.Gamma_a[ii - 1]
            self.Gamma_b[ii] = self.Gamma_b[ii - 1]
            current_tol = 1.
            counter = 0
            temp_ans = np.array([self.u_approx[ii], self.v_approx[ii], self.Gamma_a[ii], self.Gamma_b[ii]])
            if ii > 2:
                self.Gamma_a[ii] = 2 * self.Gamma_a[ii - 1] - self.Gamma_a[ii - 2]
                self.Gamma_b[ii] = 2 * self.Gamma_b[ii - 1] - self.Gamma_b[ii - 2]
            while (np.any(np.nan_to_num(current_tol) > self.etol) or counter < 2) and counter < self.max_steps:
                rr.set_initial_value([self.Gamma_a[ii-1], self.Gamma_b[ii-1]], self.tt[ii-1])
                self.u_approx[ii] = self.u_approx[ii-1]
                self.v_approx[ii] = self.v_approx[ii-1]
                u_guess = self.u_approx[ii]
                v_guess = self.v_approx[ii]
                counter_uv = 0
                while counter_uv < self.max_steps_uv:
                    u_guess = self.u_approx[ii]
                    v_guess = self.v_approx[ii]
                    self.v_approx[ii], self.u_approx[ii] = self.exact_ans_ads(self.u_approx[:ii], self.v_approx[:ii], self.tt[:ii+1], self.Gamma_a[:ii+1], self.Gamma_b[:ii+1])
                    if self.u_approx[ii] < 0:
                        self.u_approx[ii] = 0
                    if self.v_approx[ii] < 0:
                        self.v_approx[ii] = 0
                    counter_uv += 1
                u_guess = self.u_approx[ii-1]
                v_guess = self.v_approx[ii-1]
                self.Gamma_a[ii], self.Gamma_b[ii] = rr.set_f_params([v_guess, u_guess]).set_jac_params([v_guess, u_guess]).integrate(rr.t + self.dt)
                if self.u_approx[ii] < 0:
                    self.u_approx[ii] = 0
                if self.v_approx[ii] < 0:
                    self.v_approx[ii] = 0
                if self.Gamma_a[ii] < 0:
                    self.Gamma_a[ii] = 0
                if self.Gamma_b[ii] < 0:
                    self.Gamma_b[ii] = 0
                curr_ans = np.array([self.u_approx[ii], self.v_approx[ii], self.Gamma_a[ii], self.Gamma_b[ii]])
                current_tol = np.abs(curr_ans - temp_ans) / curr_ans
                temp_ans = curr_ans
                counter += 1
            self.I[ii] = self.k_func(self.delta_G_func(self.tt[ii]), 'f') * self.v_approx[ii] - self.k_func(self.delta_G_func(self.tt[ii]), 'b') * self.u_approx[ii]
            self.I_ads[ii] = self.k_func_ads(self.delta_G_func_ads(self.tt[ii]), 'f') * self.Gamma_a[ii] - self.k_func_ads(self.delta_G_func_ads(self.tt[ii]), 'b') * self.Gamma_b[ii]
            if ii % self.print_step == 0:
                print('Done with time step ', ii, ' of ', self.time_steps)#'iter_steps, c_A, c_B, Gamma_A, Gamma_B = ', counter, v_approx[ii], u_approx[ii], Gamma_a[ii], Gamma_b[ii], v_approx[ii] + u_approx[ii])
                if self.plot_T_F:
                    self.plot()

        self.Gamma_a_rev[0] = self.Gamma_a[-1]
        self.Gamma_b_rev[0] = self.Gamma_b[-1]
        u_approx_rev = np.zeros((self.time_steps))
        u_approx_rev[0] = self.u_approx[-1]
        v_approx_rev = np.zeros((self.time_steps))
        v_approx_rev[0] = self.v_approx[-1]
        self.I_rev[0] = self.I[-1]
        self.I_ads_rev[0] = self.I_ads[-1]
        rr.set_initial_value([self.Gamma_a_rev[-1], self.Gamma_b_rev[-1]], 0)
        self.p_start, self.p_end = self.p_end, self.p_start
        self.scan_direction = 1 if self.p_end > self.p_start else -1
        self.k_f_dict = {}
        self.k_b_dict = {}
        self.c_A_init = self.v_approx[-1]
        self.c_B_init = self.u_approx[-1]
        self.p_list = np.linspace(self.p_start, self.p_end, self.time_steps + 1)

        for ii in range(1, self.time_steps):
            self.Gamma_a_rev[ii] = self.Gamma_a_rev[ii - 1]
            self.Gamma_b_rev[ii] = self.Gamma_b_rev[ii - 1]
            current_tol = 1.
            counter = 0
            temp_ans = np.array([u_approx_rev[ii], v_approx_rev[ii], self.Gamma_a_rev[ii], self.Gamma_b_rev[ii]])
            if ii > 2:
                self.Gamma_a_rev[ii] = 2 * self.Gamma_a_rev[ii - 1] - self.Gamma_a_rev[ii - 2]
                self.Gamma_b_rev[ii] = 2 * self.Gamma_b_rev[ii - 1] - self.Gamma_b_rev[ii - 2]
            while (np.any(np.nan_to_num(current_tol) > self.etol) or counter < 2) and counter < self.max_steps:
                rr.set_initial_value([self.Gamma_a_rev[ii-1], self.Gamma_b_rev[ii-1]], self.tt[ii-1])
                u_approx_rev[ii] = u_approx_rev[ii-1]
                v_approx_rev[ii] = v_approx_rev[ii-1]
                u_guess = u_approx_rev[ii]
                v_guess = v_approx_rev[ii]
                counter_uv = 0
                while counter_uv < self.max_steps_uv:
                    u_guess = u_approx_rev[ii]
                    v_guess = v_approx_rev[ii]
                    v_approx_rev[ii], u_approx_rev[ii] = self.exact_ans_ads(u_approx_rev[:ii], v_approx_rev[:ii], self.tt[:ii+1], self.Gamma_a_rev[:ii+1], self.Gamma_b_rev[:ii+1])
                    if u_approx_rev[ii] < 0:
                        u_approx_rev[ii] = 0
                    if v_approx_rev[ii] < 0:
                        v_approx_rev[ii] = 0
                    counter_uv += 1
                u_guess = u_approx_rev[ii-1]
                v_guess = v_approx_rev[ii-1]
                self.Gamma_a_rev[ii], self.Gamma_b_rev[ii] = rr.set_f_params([v_guess, u_guess]).set_jac_params([v_guess, u_guess]).integrate(rr.t + self.dt)
                if u_approx_rev[ii] < 0:
                    u_approx_rev[ii] = 0
                if v_approx_rev[ii] < 0:
                    v_approx_rev[ii] = 0
                if self.Gamma_a_rev[ii] < 0:
                    self.Gamma_a_rev[ii] = 0
                if self.Gamma_b_rev[ii] < 0:
                    self.Gamma_b_rev[ii] = 0
                curr_ans = np.array([u_approx_rev[ii], v_approx_rev[ii], self.Gamma_a_rev[ii], self.Gamma_b_rev[ii]])
                current_tol = np.abs(curr_ans - temp_ans) / curr_ans
                temp_ans = curr_ans
                counter += 1
            self.I_rev[ii] = self.k_func(self.delta_G_func(self.tt[ii]), 'f') * v_approx_rev[ii] - self.k_func(self.delta_G_func(self.tt[ii]), 'b') * u_approx_rev[ii]
            self.I_ads_rev[ii] = self.k_func_ads(self.delta_G_func_ads(self.tt[ii]), 'f') * self.Gamma_a_rev[ii] - self.k_func_ads(self.delta_G_func_ads(self.tt[ii]), 'b') * self.Gamma_b_rev[ii]
            if ii % self.print_step == 0:
                print('Done with time step ', ii, 'of ', self.time_steps)#'iter_steps, c_A, c_B, Gamma_A, Gamma_B = ', counter, v_approx_rev[ii], u_approx_rev[ii], Gamma_a_rev[ii], Gamma_b_rev[ii])
                if self.plot_T_F:
                    self.plot()

        if self.plot_T_F:
            plt.close()
            fig, ax = plt.subplots()
            ax.set_xlabel("V", fontsize=24)
            ax.set_ylabel(r"$I/FAc_0 (cm^{-1}s^{-1})$", fontsize=24)
            ax.set_title("Simulated CV")
            ax.set_xlim(self.p_start * 27.211, self.p_end * 27.211)
            ax.plot(self.p_list[::-1][1:] * 27.211, self.I + self.I_ads)
            ax.plot(self.p_list[1:] * 27.211, self.I_rev + self.I_ads_rev)
            plt.show()

    def plot(self):
        # pyplot.figure((8,8))
        self.for_line.set_data(self.p_list[::-1][1:] * 27.211, self.I + self.I_ads)
        ylim_max = max(self.I + self.I_ads)*1.1
        y_lim_min = min(self.I_rev + self.I_ads_rev)*1.1
        self.ax.set_ylim(-max(ylim_max, abs(y_lim_min)), ylim_max)
        self.rev_line.set_data(self.p_list[1:] * 27.211, self.I_rev + self.I_ads_rev)
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except:
            plt.pause(1e-2)

class Formatted_Label(tk.Text, object):
    def __init__(self, master):
        super(Formatted_Label, self).__init__(master, borderwidth=0, background=master.cget("background"), pady=4)
        self['state'] = 'normal'
        self.tag_configure("subscript", offset=-4)
        self.tag_configure("subscript", font=('Serif', 8))
        self.tag_configure("superscript", offset=4)
        self.tag_configure("superscript", font=('Serif', 8))

    def insert(self, *args):
        self['state'] = 'normal'
        super(Formatted_Label, self).insert(*args)
        self.configure(width=len(self.get("1.0", "end")) - 1)
        self.configure(height=sum([1 if x == '\n' else 0 for x in self.get("1.0", "end")]) + 2 if IS_MAC else 1)
        self['state'] = 'disabled'
    
    def set(self, new):
        self['state'] = 'normal'
        self.delete('1.0', tk.END)
        self.insert("insert", new)
        self['state'] = 'disabled'
    
class MyDialog(tkd.Dialog, object):

    def __init__(self, master):
        super(MyDialog, self).__init__(master)
        self.master = master

    def body(self, master):
        # Configure main grid
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=1)
        master.columnconfigure(3, weight=1)

        # self.should_plot = tk.IntVar()
        self.isotherm = tk.StringVar()
        self.use_MH = tk.IntVar()
        self.run_adsorption = tk.IntVar()
        self.complex_rates = tk.IntVar()

        # Left side
        left_column = tk.Frame(master)
        left_column.grid(row=0, column=0, columnspan=2, sticky='new')
        left_param_background = tk.Frame(left_column, bg='blue', padx=3, pady=3)
        self.left_param_frame = tk.Frame(left_param_background, padx=10, pady=10)
        self.reorg_in = tk.Entry(self.left_param_frame)
        self.reorg_in.insert(0, REORG_DEF)
        self.gamma_in = tk.Entry(self.left_param_frame)
        self.gamma_in.insert(0, GAMMA_DEF)
        self.temp_in = tk.Entry(self.left_param_frame)
        self.temp_in.insert(0, TEMP_DEF)
        self.e6 = tk.Entry(self.left_param_frame)
        self.e6.insert(0, ALPHA_DEF)
        self.scan_temp_in = tk.Entry(self.left_param_frame)
        self.scan_temp_in.insert(0, SCAN_TEMP_DEF)
        self.e8 = tk.Entry(self.left_param_frame)
        self.e8.insert(0, TIME_STEPS_DEF)
        self.e8['state'] = 'disabled'
        self.v_start_in = tk.Entry(self.left_param_frame)
        self.v_start_in.insert(0, V_START_DEF)
        self.v_end_in = tk.Entry(self.left_param_frame)
        self.v_end_in.insert(0, V_END_DEF)

        self.left_param_frame.pack()
        left_param_background.grid(row=0, column=0, columnspan=2, sticky='new')
        tk.Label(self.left_param_frame, text="ET rate expression:").grid(row=0)
        self.gamma_label = Formatted_Label(self.left_param_frame)
        self.gamma_label.insert("insert", "")
        self.gamma_label.grid(row=1)
        self.alpha_label = tk.Label(self.left_param_frame, text=u"\u03B1:")
        self.alpha_label.grid(row=2)
        self.reorg_label = tk.Label(self.left_param_frame, text=u"\u03bb (.1eV - 2eV):")
        self.reorg_label.grid(row=3)
        temp_label = Formatted_Label(self.left_param_frame)
        temp_label.insert("insert", "Temperature (K):")
        temp_label.grid(row=4)
        nu_label = Formatted_Label(self.left_param_frame)
        nu_label.insert("insert", u"\u03bd (Volt/s):")
        nu_label.grid(row=5)
        tk.Label(self.left_param_frame, text="Number of time steps:").grid(row=6)
        v_start_label = Formatted_Label(self.left_param_frame)
        v_start_label.insert("insert", "V", "", "start", "subscript", "(Volt):")
        v_start_label.grid(row=7)
        v_end_label = Formatted_Label(self.left_param_frame)
        v_end_label.insert("insert", "V", "", "end", "subscript", "(Volt):")
        v_end_label.grid(row=8)

        self.gamma_in.grid(row=1, column=1)
        self.e6.grid(row=2, column=1)
        self.reorg_in.grid(row=3, column=1)
        self.temp_in.grid(row=4, column=1)
        self.scan_temp_in.grid(row=5, column=1)
        self.e8.grid(row=6, column=1)
        self.v_start_in.grid(row=7, column=1)
        self.v_end_in.grid(row=8, column=1)

        mhc_frame = tk.Frame(self.left_param_frame, pady=10)
        r_mhc = tk.Radiobutton(mhc_frame, text="Marcus-Hush-Chidsey", variable=self.use_MH, value=1, justify='left', command=lambda: self.handle_rate_select("MHC"))
        r_bv = tk.Radiobutton(mhc_frame, text="Butler-Volmer", variable=self.use_MH, value=0, justify='left', command=lambda: self.handle_rate_select("BV"))
        r_mhc.pack()
        r_bv.pack()
        if MH_DEF:
            r_mhc.select()
            self.handle_rate_select("MHC")
        else:
            r_bv.select()
            self.handle_rate_select("BV")
        
        mhc_frame.grid(row=0, column=1, sticky="W")

        # Right column
        right_column_padder = tk.Frame(master, pady=0)
        right_adsorb_background = tk.Frame(right_column_padder, bg='blue', padx=BG_SIZE, pady=BG_SIZE)
        self.right_adsorb_frame = tk.Frame(right_adsorb_background, padx=10, pady=10)
        right_column_padder.grid(row=0, column=2, columnspan=2, sticky='new')
        right_adsorb_background.grid(row=0, column=0, sticky="new")
        self.right_adsorb_frame.pack(fill='x')
        gamma_s_label = Formatted_Label(self.right_adsorb_frame)
        gamma_s_label.insert('insert', u'\u0393', '', 's', 'subscript', "(1/cm", "", "2", "superscript", "):")
        gamma_s_label.grid(row=2, column=0)
        gamma_ads_label = Formatted_Label(self.right_adsorb_frame)
        gamma_ads_label.insert("insert", "k", "", "0", "subscript", "(s", "", "-1", "superscript", "):")
        gamma_ads_label.grid(row=3, column=0)
        k_label = Formatted_Label(self.right_adsorb_frame)
        k_label.insert("insert", "k")
        k_label.grid(row=4, column=0)
        self.k_ads_a_label = Formatted_Label(self.right_adsorb_frame)
        self.k_ads_a_label.insert("insert", "k", "", "a", "subscript", "ads", "superscript")
        self.k_ads_a_label.grid(row=5, column=0)
        self.k_ads_b_label = Formatted_Label(self.right_adsorb_frame)
        self.k_ads_b_label.insert("insert", "k", "", "b", "subscript", "ads", "superscript")
        self.k_ads_b_label.grid(row=6, column=0)
        self.k_des_a_label = Formatted_Label(self.right_adsorb_frame)
        self.k_des_a_label.insert("insert", "k", "", "a", "subscript", "des", "superscript")
        self.k_des_a_label.grid(row=7, column=0)
        self.k_des_b_label = Formatted_Label(self.right_adsorb_frame)
        self.k_des_b_label.insert("insert", "k", "", "b", "subscript", "des", "superscript")
        self.k_des_b_label.grid(row=8, column=0)

        self.e9 = tk.Entry(self.right_adsorb_frame)
        self.e9.insert(1, GAMMA_S_DEF)
        self.e10 = tk.Entry(self.right_adsorb_frame)
        self.e10.insert(1, GAMMA_ADS_DEF)
        self.e12 = tk.Entry(self.right_adsorb_frame)
        self.e12.insert(1, K_ADS_A_DEF)
        self.e13 = tk.Entry(self.right_adsorb_frame)
        self.e13.insert(1, K_ADS_B_DEF)
        self.e14 = tk.Entry(self.right_adsorb_frame)
        self.e14.insert(1, K_DES_A_DEF)
        self.e15 = tk.Entry(self.right_adsorb_frame)
        self.e15.insert(1, K_DES_B_DEF)
        self.e16 = tk.Entry(self.right_adsorb_frame)
        self.e16.insert(1, K_ADS_A_DEF)
        
        self.e9.grid(row=2, column=1)
        self.e10.grid(row=3, column=1)
        self.e12.grid(row=5, column=1)
        self.e13.grid(row=6, column=1)
        self.e14.grid(row=7, column=1)
        self.e15.grid(row=8, column=1)
        self.e16.grid(row=4, column=1)

         # Isotherm box frame
        isotherm_padder = tk.Frame(right_column_padder, pady=10)
        isotherm_background = tk.Frame(isotherm_padder, bg='blue', padx=3, pady=3)
        isotherm_frame = tk.Frame(isotherm_background, padx=10, pady=10)
        tk.Label(isotherm_frame, text="Isotherm:").grid(row=0)
        self.isotherm_params_label = tk.Label(isotherm_frame, text="Parameters")
        self.isotherm_params_label.grid(row=1)
        self.isotherm_param_label_a = Formatted_Label(isotherm_frame)
        self.isotherm_param_label_a.insert("insert", "")
        self.isotherm_param_label_b = Formatted_Label(isotherm_frame)
        self.isotherm_param_label_b.insert("insert", "")
        self.isotherm_param_label_a.grid(row=2)
        self.isotherm_param_label_b.grid(row=3)

        self.e1 = tk.Entry(isotherm_frame)
        self.e2 = tk.Entry(isotherm_frame)

        radio_frame = tk.Frame(isotherm_frame)
        self.r1 = tk.Radiobutton(radio_frame, text='Langmuir', variable=self.isotherm, value="Langmuir", command=lambda: self.handleIsothermChange("Langmuir"))
        self.r2 = tk.Radiobutton(radio_frame, text='Frumkin', variable=self.isotherm, value="Frumkin", command=lambda: self.handleIsothermChange("Frumkin"))
        # r3 = tk.Radiobutton(radio_frame, text='Temkin', variable=self.isotherm, value="Temkin", command=lambda: self.handleIsothermChange("Temkin"))
        self.r1.pack()
        self.r2.pack()
        # r3.pack()
        radio_frame.grid(row=0, column=1)
        self.e1.grid(row=2, column=1, sticky="E")
        self.e2.grid(row=3, column=1, sticky="E")
        if ISOTHERM_DEF == "Langmuir":
            self.r1.select()
        elif ISOTHERM_DEF == "Frumkin":
            self.r2.select()
        # else:
        #     r3.select()
        self.handleIsothermChange(ISOTHERM_DEF)
        isotherm_frame.columnconfigure(1, weight=2)
        isotherm_frame.pack(fill='x')
        isotherm_padder.grid(row=1, sticky='new')
        isotherm_background.pack(fill='both')

        # Different k toggle
        complex_rates_label = tk.Label(self.right_adsorb_frame, text="Complex rates?: ")
        complex_rates_label.grid(row=1)
        complex_rates_label.grid_configure(pady=20)
        self.complex_rates_toggle = tk.Checkbutton(self.right_adsorb_frame, variable=self.complex_rates, justify='left', command=self.handle_complex_rates_toggle)
        self.complex_rates_toggle.grid(row=1, column=1)
        self.complex_rates_toggle.grid_configure(pady=20)
        self.handle_complex_rates_toggle()

        # Adsorption toggle
        use_ads_label = tk.Label(self.right_adsorb_frame, text="Use adsorption?:")
        use_ads_label.grid(row=0)
        use_ads_label.grid_configure(pady=20)
        adsorption_toggle = tk.Checkbutton(self.right_adsorb_frame, variable=self.run_adsorption, justify='left', command=self.handle_adsorption_toggle)
        adsorption_toggle.grid(row=0, column=1)
        adsorption_toggle.grid_configure(pady=20)
        if ADS_DEF:
            adsorption_toggle.select()
        self.handle_adsorption_toggle()

        # Read from file menu
        # file_button = tk.Button(master, text='Read/run from file', command=self.open_file)

        # Help button
        # help_button = tk.Button(master, text="Help", command=self.show_help)

        # Plot?
        # self.e19 = tk.Checkbutton(master, variable=self.should_plot, justify='left')
        # if PLOT_T_F_DEF:
        #     self.e19.select()
        
        # tk.Label(master, text="Plot?:").grid(column=1, row=3, sticky="E")
        # self.e19.grid(row=3, column=2, sticky="W")
        # file_button.grid(row=4, column=1, columnspan=2)
        # help_button.grid(row=5, column=1, columnspan=2)

        for child in master.winfo_children():
            child.grid_configure(pady=5, padx=15)
        
        return self.e1 # initial focus
    
    def open_file(self):
        file = askopenfilename()
        if file != () and file != None:
            self.destroy()
            sim = CV_Simulator(None, file)
            sim.run()
            if sim.plot_T_F:
                sim.plot()
    
    def handleIsothermChange(self, isotherm):
        self.e1.delete(0, len(self.e1.get()))
        self.e2.delete(0, len(self.e2.get()))
        self.e1.grid()
        self.e2.grid()
        self.isotherm_param_label_a.grid()
        self.isotherm_param_label_b.grid()
        self.isotherm_params_label.grid()
        if isotherm == "Frumkin":
            self.isotherm_param_label_a.set("")
            self.isotherm_param_label_a.insert("insert", "g", "", "a", "subscript", "(-.002 - .002)", "")
            self.isotherm_param_label_b.set("")
            self.isotherm_param_label_b.insert("insert", "g", "", "b", "subscript", "(-.002 - .002)", "")

            self.e1['state'] = 'normal'
            self.e2['state'] = 'normal'
            self.e1.insert(0, G_FRUM_A_DEF)
            self.e2.insert(0, G_FRUM_B_DEF)
        elif isotherm == "Temkin":
            self.isotherm_param_label_a.set("cov:")
            self.isotherm_param_label_b.set("char:")
            self.e1['state'] = 'normal'
            self.e2['state'] = 'normal'
            self.e1.insert(0, COV_DEF)
            self.e2.insert(0, CHAR_DEF)
        else:
            self.isotherm_param_label_a.set("N/A")
            self.isotherm_param_label_b.set("N/A")
            self.e1['state'] = 'disabled'
            self.e2['state'] = 'disabled'
            self.e1.grid_remove()
            self.e2.grid_remove()
            self.isotherm_param_label_a.grid_remove()
            self.isotherm_param_label_b.grid_remove()
            self.isotherm_params_label.grid_remove()
    
    def handle_adsorption_toggle(self):
        if self.run_adsorption.get() == 0:
            self.e9['state'] = 'disabled'
            self.e10['state'] = 'disabled'
            self.e12['state'] = 'disabled'
            self.e13['state'] = 'disabled'
            self.e14['state'] = 'disabled'
            self.e15['state'] = 'disabled'
            self.e16['state'] = 'disabled'
            self.complex_rates_toggle['state'] = 'disabled'

            self.r1['state'] = 'disabled'
            self.r2['state'] = 'disabled'
            self.e1['state'] = 'disabled'
            self.e2['state'] = 'disabled'

            self.gamma_in['state'] = 'normal'
        else:
            self.e9['state'] = 'normal'
            self.e10['state'] = 'normal'
            self.e12['state'] = 'normal'
            self.e13['state'] = 'normal'
            self.e14['state'] = 'normal'
            self.e15['state'] = 'normal'
            self.e16['state'] = 'normal'
            self.complex_rates_toggle['state'] = 'normal'
            
            self.r1['state'] = 'normal'
            self.r2['state'] = 'normal'
            self.e1['state'] = 'normal'
            self.e2['state'] = 'normal'

            self.gamma_in.delete(0, tk.END)
            self.gamma_in.insert(0, '0')
            self.gamma_in['state'] = 'disabled'
        
        self.handle_complex_rates_toggle()
    
    def handle_complex_rates_toggle(self):
        if self.complex_rates.get() == 0:
            self.e12['state'] = 'disabled'
            self.e13['state'] = 'disabled'
            self.e14['state'] = 'disabled'
            self.e15['state'] = 'disabled'
            self.e12.grid_remove()
            self.e13.grid_remove()
            self.e14.grid_remove()
            self.e15.grid_remove()
            self.k_ads_a_label.grid_remove()
            self.k_ads_b_label.grid_remove()
            self.k_des_a_label.grid_remove()
            self.k_des_b_label.grid_remove()
            if self.run_adsorption.get() == 1:
                self.e16['state'] = 'normal'
            
        else:
            self.e16['state'] = 'disabled'
            if self.run_adsorption.get() == 1:
                self.e12['state'] = 'normal'
                self.e13['state'] = 'normal'
                self.e14['state'] = 'normal'
                self.e15['state'] = 'normal'
                self.e12.grid()
                self.e13.grid()
                self.e14.grid()
                self.e15.grid()
                self.k_ads_a_label.grid()
                self.k_ads_b_label.grid()
                self.k_des_a_label.grid()
                self.k_des_b_label.grid()

    
    def handle_rate_select(self, expression):
        if expression == "MHC":
            self.reorg_label.grid()
            self.reorg_in.grid()
            self.alpha_label.grid_remove()
            self.e6.grid_remove()
            self.gamma_label.set("")
            self.gamma_label.insert("insert", "k", "", "0", "subscript", "MHC", "superscript", "(.01s", "", "-1", "superscript", " - 10", "", "4", "superscript", "s", "", "-1", "superscript", "):")
            self.gamma_in.delete(0, tk.END)
            self.gamma_in.insert(0, '100')
        else:
            self.alpha_label.grid()
            self.e6.grid()
            self.reorg_label.grid_remove()
            self.reorg_in.grid_remove()
            self.gamma_label.set("")
            self.gamma_label.insert("insert", "k", "", "0", "subscript", "BV", "superscript", "(.01s", "", "-1", "superscript", " - 10", "", "s", "", "-1", "superscript", "):")
            self.gamma_in.delete(0, tk.END)
            self.gamma_in.insert(0, '10')

    def show_help(self):
        help_window = tk.Toplevel()
        help_window.wm_title("Help window")

        description_title = tk.Label(help_window, text="Program description")
        description_title.pack()

        description_body = tk.Text(help_window, wrap=tk.WORD, height=5, padx=10)
        description_body.insert("insert", 
        """
        GUI program to simulate cyclic voltammetry using a grid-free approach\n
        See https://doi.org/10.1063/5.0044156
        """)
        description_body.pack()

        parameters_title = tk.Label(help_window, text='Parameter descriptions')
        parameters_title.pack()
        parameters_body = tk.Text(help_window, wrap=tk.WORD, spacing3=15, height=16, padx=10)
        parameters_body.tag_configure("subscript", offset=-4, font=('Serif', 6))
        parameters_body.insert("insert", "- Isotherm: adsorption isotherm model to use in simulation. Langmuir has no additional parameters, Frumkin has two. For more information on adsorption isotherms, see https://en.wikipedia.org/wiki/Adsorption#Langmuir. For more information on available isotherms, see above paper.\n")
        #FIXME: unicode character should now be $\lambda$
        parameters_body.insert("insert", u"- \u03A9: Reorganization energy for Marcus-Hush-Chidsey (MHC) dynamics\n")
        parameters_body.insert("insert", u"- \u0393: Standard rate constant for either Butler-Volmer (BV) or Marcus-Hush-Chidsey (MHC) dynamics\n")
        parameters_body.insert("insert", u"- \u03B1: Charge transfer coefficient for Butler-Volmer (BV) dynamics\n")
        #FIXME: scan rate should be unicode for $\nu$
        parameters_body.insert("insert", "- scan rate: Rate at which simulated potential is ramped, in Volts per second\n")
        parameters_body.insert("insert", "- time steps: Number of time steps in the forward and backward direction.\n")
        #FIXME: UNITS should be mol/cm^{2}, with correct formatting
        parameters_body.insert("insert", u"- \u0393\u209B: Saturated surface concentration in UNITS\n")
        #FIXME: Gamma ads should be $k_{0}^{ads}$
        parameters_body.insert("insert", u"- \u0393")
        parameters_body.insert("insert", "ads", "subscript")
        parameters_body.insert("insert", ": Standard rate constant for adsorbed electron transfer\n")
        parameters_body.insert("insert", "- k")
        parameters_body.insert("insert", "ads", "subscript")
        parameters_body.insert("insert", "/k")
        parameters_body.insert("insert", "des", "subscript")
        parameters_body.insert("insert", ": Rate constant of adsorption/desorption of species, in inverse seconds a/b\n")
        parameters_body.insert("insert", "- V", "", "start", "subscript", "/V", "", "end", "subscript", ": Range of driving voltages\n")
        parameters_body.insert("insert", "- MHC/BV: Expression to use for the electron transfer rate constant (Marcus-Hush-Chidsey or Butler-Volmer)\n")
        parameters_body.pack()

        from_file_title = tk.Label(help_window, text='Running from file')
        from_file_title.pack()
        from_file_body = tk.Text(help_window, wrap=tk.WORD, padx=10)
        from_file_body.insert("insert", "To run simulation from file, use gui and select \"read/run from file\" or run script from command line with --file flag. File should be formatted as python dictionary, and contain all the parameters listed above.")
        from_file_body.pack()

    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        try:
            self.apply()
        except:
            pass
    
    def apply(self):
        param_dict = {}
        # Optionally set parameters from selected isotherm
        if self.isotherm.get() == 'Frumkin':
            param_dict['g_frum_a'] = float(self.e1.get())
            param_dict['g_frum_b'] = float(self.e2.get())
        elif self.isotherm.get() == 'Temkin':
            param_dict['cov'] = float(self.e1.get())
            param_dict['char'] = float(self.e2.get())
        
        # Optionally set parameters for rate expression
        if self.use_MH.get() == 1:
            param_dict['reorg'] = float(self.reorg_in.get())
        else:
            param_dict['alpha'] = float(self.e6.get())
        
        param_dict['gamma'] = float(self.gamma_in.get())
        param_dict['temp'] = float(self.temp_in.get())
        param_dict['scan_temp'] = float(self.scan_temp_in.get())
        param_dict['time_steps'] = int(self.e8.get())
        param_dict['gamma_s'] = float(self.e9.get())
        param_dict['gamma_ads'] = float(self.e10.get())
        if self.complex_rates.get() == 1:
            param_dict['k_ads_a'] = float(self.e12.get())
            param_dict['k_ads_b'] = float(self.e13.get())
            param_dict['k_des_a'] = float(self.e14.get())
            param_dict['k_des_b'] = float(self.e15.get())
        else:
            param_dict['k_ads_a'] = float(self.e16.get())
            param_dict['k_ads_b'] = float(self.e16.get())
            param_dict['k_des_a'] = float(self.e16.get())
            param_dict['k_des_b'] = float(self.e16.get())

        param_dict['v_start'] = float(self.v_start_in.get())
        param_dict['v_end'] = float(self.v_end_in.get())
        param_dict['mh'] = bool(int(self.use_MH.get()))
        param_dict['plot_T_F'] = bool(PLOT_T_F_DEF)
        param_dict['isotherm'] = self.isotherm.get()

        self.master.quit()
        self.master.destroy()
        if self.run_adsorption.get() == 1:
            sim = CV_Simulator(param_dict)
            sim.run()
        else:
            sim = CV_No_Ads(param_dict)
            sim.run()

#Gamma_list = np.logspace(-15,3,109)
#omega_list = np.sqrt(np.linspace(0.2, 2.2, 6) / (27.211 * mass * 2 * y_A * y_A))

#Gamma_ads_list = np.logspace(-12,-5, 8)
#k_ads_list = np.logspace(-12, -5, 8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid-free cyclic voltammetry simulation")
    parser.add_argument('-f', '--file', type=str, dest='file',
        help='File from which to read parameters for simulation. If --file argument is set, script will skip gui and run straight from given file')

    args = parser.parse_args()
    # If file passed as cmd line argument, run sim, else open gui
    if (args.file != None):
        sim = CV_Simulator(None, args.file)
        sim.run()
    else:
        root = tk.Tk()
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        default_font = tkFont.nametofont("TkTextFont")
        default_font.configure(size=12)
        default_font = tkFont.nametofont("TkFixedFont")
        default_font.configure(size=12)
        root.withdraw()
        root.winfo_toplevel().title("Grid-free PyCV")
        d = MyDialog(root)
