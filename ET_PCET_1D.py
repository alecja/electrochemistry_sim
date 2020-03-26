import sys, os, time, numpy as np, math
import scipy, _pickle as cPickle, scipy.special
from math import factorial as fact
import scipy.integrate as integrate
import matplotlib.pyplot as pyplot
import pandas as pd

def fermi(y, delta_G):
    return 1. / (1. + np.exp(E(y, delta_G) / kT))

def V_B(y, delta_G):
    return 0.5 * mass * omega * omega * (y_B - y) ** 2 + delta_G

def V_A(y):
    return 0.5 * mass * omega * omega * (y_A - y) ** 2

def E(y, delta_G):
    return V_B(y, delta_G) - V_A(y)

def FC_factor(state_A, state_B):
    p = min(state_A, state_B)
    q = max(state_A, state_B)
    LP = scipy.special.genlaguerre(p, q-p)
    return (np.sqrt(fact(p) / fact(q)) * (reorg_p ** (q - p)) * np.exp(-reorg_p * reorg_p / 2) * LP(reorg_p * reorg_p) * (np.sign(p-q) ** np.abs(p - q))) ** 2

def marg_func_A(y):
    return np.exp(-(V_A(y) - V_A(y_A)) / kT)

def marg_func_B(y, delta_G):
    return np.exp(-(V_B(y, delta_G) - V_B(y_B, delta_G)) / kT)

def int_func_A(y, *args):
    delta_G = args[0]
    return marg_func_A(y) * fermi(y, delta_G)

def int_func_B(y, *args):
    delta_G = args[0]
    return marg_func_B(y, delta_G) * (1. - fermi(y, delta_G))

def k_func(energy, dir, state_A, state_B):
    if dir == 'f':
        if MH:
            return Gamma * FC_factor(state_A, state_B) * integrate.quad(int_func_A, -np.inf, np.inf, args=(energy,))[0] / (dx * integrate.quad(marg_func_A, -np.inf, np.inf)[0])
        else:
            return Gamma * FC_factor(state_A, state_B) * np.longdouble(np.exp(-alpha * epsilon * energy)) / dx
    if dir == 'b':
        if MH:
            return Gamma * FC_factor(state_A, state_B) * integrate.quad(int_func_B, -np.inf, np.inf, args=(energy,))[0] / (dx * integrate.quad(marg_func_B, -np.inf, np.inf, args=(energy,))[0])
        else:
            return Gamma * FC_factor(state_A, state_B) * np.longdouble(np.exp((1. - alpha) * epsilon * energy)) / dx

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

stencil = '3'
zero_coeff = -2.
one_coeff = 1.
two_coeff = 0.
three_coeff = 0.
four_coeff = 0.
five_coeff = 0.
six_coeff = 0.
seven_coeff = 0.
coeff_array = np.array([zero_coeff,one_coeff,two_coeff,three_coeff,four_coeff,five_coeff,six_coeff,seven_coeff])

def make_matrix(delta_G):
    L_A = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_A))
    L_B = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_B))
    eta_A = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_A))
    eta_B = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_B))
    eta_A_tilde = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_A))
    eta_B_tilde = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_B))
    eta_A_diag = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_A))
    eta_B_diag = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_B))
    U_A = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_A))
    U_B = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_B))
    U_A_inv = np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_A))
    U_B_inv = np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_B))
    for ii in range(grid_pts_x * vib_states_A):
        for jj in range(grid_pts_x * vib_states_A):
            if jj == ii + 1 or jj == ii - 1:
                L_A[ii,jj] = one_coeff * D_A / (dx * dx)
            if jj == ii:
                L_A[ii,jj] = zero_coeff * D_A / (dx * dx)
    for ii in range(grid_pts_x * vib_states_B):
        for jj in range(grid_pts_x * vib_states_B):
            if jj == ii + 1 or jj == ii - 1:
                L_B[ii,jj] = one_coeff * D_B / (dx * dx)
            if jj == ii:
                L_B[ii,jj] = zero_coeff * D_B / (dx * dx)
    for jj in range(vib_states_A):
        for kk in range(vib_states_B):
            state_A = jj
            state_B = kk
            energy = delta_G + omega_p * (kk - jj)
            eta_A_diag[grid_pts_x * jj, grid_pts_x * jj] += (k_func(energy, 'f', state_A, state_B))
            eta_B_diag[grid_pts_x * kk, grid_pts_x * kk] += (k_func(energy, 'b', state_A, state_B))
            eta_A[grid_pts_x * kk, grid_pts_x * jj] = k_func(energy, 'f', state_A, state_B)
            eta_A_tilde[grid_pts_x * kk, grid_pts_x * jj] = eta_A[grid_pts_x * kk, grid_pts_x * jj] * np.exp(energy * epsilon / 2)
            eta_B[grid_pts_x * jj, grid_pts_x * kk] = k_func(energy, 'b', state_A, state_B)
            eta_B_tilde[grid_pts_x * jj, grid_pts_x * kk] = eta_B[grid_pts_x * jj, grid_pts_x * kk] * np.exp(-energy * epsilon / 2)
    mesh = int(int(stencil) / 2)
    for ii in range(mesh):
        for jj in range(vib_states_A):
            L_A[ii + (jj * grid_pts_x),ii + (jj * grid_pts_x)] += np.sum(coeff_array[ii+1:]) * D_A / (dx * dx)
            if jj > 0:
                L_A[ii + (jj * grid_pts_x) - 1,ii + (jj * grid_pts_x)] = 0
                L_A[ii + (jj * grid_pts_x),ii + (jj * grid_pts_x) - 1] = 0
        for jj in range(vib_states_B):
            L_B[ii + (jj * grid_pts_x),ii + (jj * grid_pts_x)] += np.sum(coeff_array[ii+1:]) * D_B / (dx * dx)
            if jj > 0:
                L_B[ii + (jj * grid_pts_x) - 1,ii + (jj * grid_pts_x)] = 0
                L_B[ii + (jj * grid_pts_x),ii + (jj * grid_pts_x) - 1] = 0
    for ii in range(vib_states_A):
        U_A[ii * grid_pts_x:(ii + 1) * grid_pts_x,ii * grid_pts_x:(ii + 1) * grid_pts_x] = np.diag(np.tile(np.exp((omega_p * (ii)) / (2. * kT)), grid_pts_x))
        U_A_inv[ii * grid_pts_x:(ii + 1) * grid_pts_x,ii * grid_pts_x:(ii + 1) * grid_pts_x] = np.diag(np.tile(np.exp(-(omega_p * (ii)) / (2. * kT)), grid_pts_x))
    for ii in range(vib_states_B):
        U_B[ii * grid_pts_x:(ii + 1) * grid_pts_x,ii * grid_pts_x:(ii + 1) * grid_pts_x] = np.diag(np.tile(np.exp((delta_G + omega_p * (ii)) / (2. * kT)), grid_pts_x))
        U_B_inv[ii * grid_pts_x:(ii + 1) * grid_pts_x,ii * grid_pts_x:(ii + 1) * grid_pts_x] = np.diag(np.tile(np.exp(-(delta_G  + omega_p * (ii)) / (2. * kT)), grid_pts_x))
    M = np.bmat([[L_A - eta_A_diag, eta_B], [eta_A, L_B - eta_B_diag]]).A
    k_f_array = np.zeros((vib_states_A))
    k_b_array = np.zeros((vib_states_B))
    k_f_array = np.diag(eta_A_diag)[::grid_pts_x]
    k_b_array = np.diag(eta_B_diag)[::grid_pts_x]
    M_tilde = np.bmat([[L_A - eta_A_diag, eta_B_tilde], [eta_A_tilde, L_B - eta_B_diag]]).A
    U = np.bmat([[U_A, np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_B))], [np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_A)), U_B]]).A
    U_inv = np.bmat([[U_A_inv, np.zeros((grid_pts_x * vib_states_A, grid_pts_x * vib_states_B))], [np.zeros((grid_pts_x * vib_states_B, grid_pts_x * vib_states_A)), U_B_inv]]).A
    eigvalue, eigvector = np.linalg.eigh(M_tilde)
    eigvalue_inv = eigvalue ** -1
    return M_tilde, U, U_inv, eigvalue, eigvector, eigvalue_inv

def prop_matrix(C_alt, M_tilde, U, U_inv, eigvalue, eigvector, eigvalue_inv, dt):
    #b = np.zeros((grid_pts_x * (vib_states_A + vib_states_B)))
    b = np.dot(np.dot(np.transpose(eigvector), U), C_alt)
    y = np.zeros((grid_pts_x * (vib_states_A + vib_states_B)))
    v = np.dot(U, C_alt)
    for ii in range(vib_states_A + vib_states_B):
        y[grid_pts_x * (ii + 1) - 1] = -np.dot(M_tilde[grid_pts_x * (ii + 1) - 1,:], v)
    z = np.dot(np.transpose(eigvector), y)
    b_eq = -z * eigvalue_inv
    b = np.longdouble(np.exp(eigvalue * dt)) * (b - b_eq) + b_eq
    C_alt = np.dot(np.dot(U_inv, eigvector), b)
    C_alt[C_alt < 0] = 0
    I_SME = D * np.sum((C_alt[1:grid_pts_x * vib_states_A][::grid_pts_x] - C_alt[:grid_pts_x * vib_states_A][::grid_pts_x])) / dx
    if len(eigvalue[eigvalue >0]) > 0:
            print('Positive eigenvalues = ', eigvalue[eigvalue >0])
            print('Positive eigenvalue at step ', tt)
    return C_alt, I_SME

np.set_printoptions(linewidth=300)

#ET or PCET
PCET = False
#MHC rate or BV rate
MH = True
#Save to output
save_data = True

###Parameters for simulation

#Size of grid (x = diffusion, y = solvent reorganization)
grid_pts_x = 50
grid_pts_y = 101

#Starting and ending overpotential
P_start = 0.05
P_end = -0.05

#ET parameters (in a.u.)
mass = 2000
omega = 0.00025
Gamma = 1e-5
kT = 0.00095
epsilon = 1 / kT
alpha = 0.5
well = 10
y_B = well
y_A = -well
ymin = -4 * well
ymax = 4 * well

#Simulation parameters
scan_rate = 1e-16
D = D_A = D_B = 0.000475

#Parameters for proton
if not PCET:
    omega_p = 0
    reorg_p = 0
    vib_states_A = 1
    vib_states_B = 1
else:
    omega_p = 0.012
    reorg_p = 1.
    vib_states_A = 2
    vib_states_B = 2

#Steps in simulation
time_steps = 2001
dt = abs(P_start - P_end) / (time_steps * scan_rate)
P_list = np.linspace(P_start, P_end, time_steps)
dx = 6 * np.sqrt(np.abs(P_start - P_end) * D / scan_rate) / grid_pts_x

#Initialize concentration array (boltzmann in y, uniform in x)
C = np.zeros((grid_pts_x * (vib_states_A + vib_states_B)))
for ii in range(vib_states_A):
    C[grid_pts_x * ii:grid_pts_x * (ii + 1)] = np.exp(-(ii) * omega_p * epsilon) / np.sum(np.exp(-np.linspace(0,vib_states_A-1,vib_states_A) * omega_p * epsilon))
I = np.zeros(time_steps)

#Propagate simulation in time
for tt in range(time_steps):
    delta_G = P_list[tt]
    M_tilde, U, U_inv, eigvalue, eigvector, eigvalue_inv = make_matrix(delta_G)
    C, I[tt] = prop_matrix(C, M_tilde, U, U_inv, eigvalue, eigvector, eigvalue_inv, dt)
    print('Done with time step ', tt + 1)

pyplot.plot(P_list, I, alpha=0.7)
pyplot.xlim(P_start, P_end)
pyplot.ylabel(r'$I(t)$', fontsize=14)
pyplot.xlabel(r'$\Delta G$', fontsize=14)
pyplot.show()

if save_data:
    df_C = pd.DataFrame(C)
    df_I = pd.DataFrame(np.array([P_list, I])).T
    df_C.to_excel(excel_writer = 'conc_results.xlsx')
    df_I.to_excel(excel_writer = 'IV_results.xlsx')
