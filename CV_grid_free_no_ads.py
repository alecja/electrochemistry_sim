import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy

class CV_No_Ads:
    def __init__(self, plot_T_F):
        self.MH = True
        reorg = 1.0
        self.mass = 2000
        self.y_A = 10
        self.y_B = -10
        self.omega = np.sqrt(reorg / (27.211 * self.mass * 2 * self.y_A * self.y_A))
        self.Gamma = 1e2
        self.kT = 0.00095# T * 8.617e-5 / 27.211
        self.D = 1e-5 #5.29e-8 * 5.29e-8 / 2.42e-17
        self.epsilon = 1. / self.kT
        self.alpha = 0.5
        self.scan_rate = 1# 1.602e-19 / 2.42e-17
        self.time_steps = 401
        self.c_A_init = 1
        self.c_B_init = 1 - self.c_A_init

        self.P_start = 0.02
        self.E_0_ads = 0
        self.P_end = -0.02
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

        # Plotting initialization
        self.plot_T_F = plot_T_F
        if self.plot_T_F:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("V")
            self.ax.set_ylabel("I/A")
            self.ax.set_title("Simulated CV (No adsorption)")
            self.ax.set_xlim(self.P_start, self.P_end)
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

<<<<<<< HEAD
    def delta_G_func(self, t):
        if type(t) != type(np.array([1])):
            result_array = np.zeros((1))
            t_list = np.array([t])
        else:
            result_array = np.zeros(len(t))
            t_list = t
        for ii, t in enumerate(t_list):
            if t < self.tt[-1]:
                result_array[ii] = self.P_start - self.scan_rate * t + self.E_0_ads
            else:
                result_array[ii] = self.P_end + self.scan_rate * (t - self.tt[-1]) + self.E_0_ads
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
=======
def delta_G_func(t):
    if type(t) != type(np.array([1])):
        result_array = np.zeros((1))
        t_list = np.array([t])
    else:
        result_array = np.zeros(len(t))
        t_list = t
    for ii, t in enumerate(t_list):
        if t < tt[-1]:
            result_array[ii] = P_start - scan_rate * t
        else:
            result_array[ii] = P_end + scan_rate * (t - tt[-1])
    return result_array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def a(t):
    try:
        if t > tt[-1]:
            ind = np.where(np.isclose(tt, t % tt[-1]))[0][0]
            return k_func(delta_G_func(t), 'b') + k_func(delta_G_func(t), 'f')
>>>>>>> 91fbaf2fa13adb0363c2d47872fe095c9f7600a8
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
<<<<<<< HEAD
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
            if ii % 50 == 0:
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
            if ii % 50 == 0:
                print('Done with time step ', ii, 'of ', self.time_steps)#'iter_steps, c_B, Gamma_A, Gamma_B = ', u_approx_rev[ii])
                if self.plot_T_F:
                    self.plot()
        
        if self.plot_T_F:
            self.plot()
            # fig, ax = plt.subplots()
            # ax.plot(self.P_list, self.I, label='sol', color='b')
            # ax.set_xlim(self.P_start * 0.99, self.P_end * 0.99)
            # ax.plot(self.P_list[::-1], self.I_rev, label='sol_rev', color='r')
            plt.show()
        
    
    def plot(self):
        #pyplot.figure((8,8))
        self.for_line.set_data(self.P_list, self.I)
        ylim = max(self.I)*1.1
        self.ax.set_ylim(-ylim, ylim)
        self.rev_line.set_data(self.P_list[::-1], self.I_rev)
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except:
            plt.pause(1e-10)

if __name__ == "__main__":
    sim = CV_No_Ads(True)
    sim.run()

#pyplot.plot(P_list, 1. - u_approx, label=r'$c_{A}$'), pyplot.plot(P_list, u_approx, label=r'$c_{B}$')
#pyplot.plot(P_list[::-1], 1. - u_approx_rev, label=r'$c^{rev}_{A}$'), pyplot.plot(P_list[::-1], u_approx_rev, label=r'$c^{rev}_{B}$')
#pyplot.xlabel(r'$\Delta G$', fontsize=16), pyplot.xlim(P_start, P_end), pyplot.legend(), pyplot.show()

=======
                result_array[ii] = Gamma * np.longdouble(np.exp((1. - alpha) * epsilon * energy))
    if len(result_array) == 1:
        return result_array[0]
    else:
        return result_array

def exact_approx(u, tt):
    ds = tt[1] - tt[0]
    t = tt[-1]
    u = np.array(u)
    v = 1. - u
    return np.longdouble((np.sqrt(D * np.pi) - a(t) * (ds * (0.5 / np.sqrt(t) + np.sum(1. / np.sqrt(t - tt[1:-1]))) - 2 * np.sqrt(t))) ** -1) * np.longdouble(ds * (np.sum((-a(tt[1:-1]) * u[1:] + b(tt[1:-1], v[:-1]) - b(t, v[-1])) / np.sqrt(t - tt[1:-1])) + 0.5 * (b(0, v[0]) - b(t, v[-1])) / np.sqrt(t)) + 2. * b(t, v[-1]) * np.sqrt(t) + c_B_init * np.sqrt(D * np.pi)).flatten()[0]

import warnings
warnings.filterwarnings("ignore")

MH = False
reorg = 0.6
mass = 2000
y_A = 10
y_B = -10
# User should now enter reorg instead of omega, units of reorg are eV and should have symbol $\lambda$
# Range of reorg is (0.1-2.0 eV). Should only be entered for MHC
omega = np.sqrt(reorg / (27.211 * mass * 2 * y_A * y_A))
reorg = 0.6
# Gamma should now be called k_{0}^{MHC} or k_{0}^{BV}, depending on the ET theory used
# The units are s^-1, and the range is (1e-2 - 1e4 s^-1)
Gamma = 1e0
# The user will enter temperature in units of Kelvin (with variable name temp), which will then be converted to kT here
# No range needed
temp = 298.
kT = 3.18e-06 * temp
# D is in units of cm^{2} / sec, and has range of (1e-6,1e-4)
D = 1e-5
epsilon =  1. / kT
# Alpha for BV ET, should be $\alpha$ in the GUI, unitless, and range (0-1). Should only be entered for BV
alpha = 0.5
#scan_rate is in units of V/sec, and user should be entering value for scan_temp. The symbol for scan_rate in the GUI should be $\nu$
#range of scan_rate is (0.0001 - 1e4) V/sec
scan_temp = 1.
scan_rate = scan_temp / 27.211
time_steps = 401
c_A_init = 1
c_B_init = 1 - c_A_init

#P_list are the driving forces/voltages; user will enter V_start and V_end (in GUI as V_{start} and V_{end})
#No range needed
V_start = 1.0
V_end = -1.0
P_start = V_start / 27.211
P_end = V_end / 27.211
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

pyplot.plot(P_list * 27.211, I, color='b')
pyplot.xlim(P_start * 27.211, P_end * 27.211)
pyplot.plot(P_list[::-1] * 27.211, I_rev, color='r')
pyplot.xlabel(r'V', fontsize=24)
pyplot.ylabel(r'I', fontsize=24)
pyplot.show()
>>>>>>> 91fbaf2fa13adb0363c2d47872fe095c9f7600a8
