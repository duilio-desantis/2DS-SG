import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import os
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import pickle
import matplotlib.ticker as ticker
from scipy.special import gamma
import matplotlib.colors as mcolors

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rcParams.update({'font.size': 27.5})
plt.rcParams['text.usetex'] = True

wd = os.getcwd()

# Custom Color Map
cmap_2Dmaps = matplotlib.colors.LinearSegmentedColormap.from_list("cmap_2Dmaps", ["white", "teal", "darkviolet"], N = 100, gamma = 1)

# I1 integral (T = 0)
def I1(w, Delta_eq, eta, beta):
    w_tilde = w/np.sqrt(2*(beta**2)*Delta_eq)
    eta_tilde = eta/np.sqrt(2*(beta**2)*Delta_eq)
    
    return ((beta**2)/(8*np.pi))*np.sqrt(-(eta_tilde - 1j*w_tilde)**2/(1 + (eta_tilde - 1j*w_tilde)**2))*np.arccos(np.sqrt(1 + (eta_tilde - 1j*w_tilde)**2))/((eta_tilde - 1j*w_tilde)**2)

# I2 integral (T = 0)
def I2(wa, wb, Delta_eq, eta, beta):
    wb += 1e-7
    
    wa_tilde = wa/np.sqrt(2*(beta**2)*Delta_eq)
    wb_tilde = wb/np.sqrt(2*(beta**2)*Delta_eq)
    eta_tilde = eta/np.sqrt(2*(beta**2)*Delta_eq)
    
    return ((beta**2)/(16*np.pi))*(1/(2*(wa_tilde - wb_tilde)*(wa_tilde + wb_tilde + 1j*2*eta_tilde))) * \
        (2*np.arccos(np.sqrt(1 + (eta_tilde - 1j*wa_tilde)**2))/(np.sqrt(-(1 + (eta_tilde - 1j*wa_tilde)**2)*(eta_tilde - 1j*wa_tilde)**2)) - \
        2*np.arccos(np.sqrt(1 + (eta_tilde - 1j*wb_tilde)**2))/(np.sqrt(-(1 + (eta_tilde - 1j*wb_tilde)**2)*(eta_tilde - 1j*wb_tilde)**2)))

# Equilibrium fluctuations (T = 0)
def D0(k, Delta_eq, beta):              
    return 1/(2*np.sqrt(k**2 + 0.5*(beta**2)*((2*np.pi)**2)*Delta_eq))

# Shorthand, employed in chi2
def Q(wa, wb, eta):
    return 1 + (eta - 1j*wa)/(eta - 1j*(wa + wb))

# Nonlinear response function (symmetrized, k-resolved)
def chi2(w1, w2, k, Delta_eq, eta, beta):
    return 0.5*(beta**2)*D0(k, Delta_eq, beta)/((2*np.pi*(w1 + w2) + 1j*2*np.pi*eta)**2 - 4*(k**2 + 0.5*(beta**2)*((2*np.pi)**2)*Delta_eq)) * \
        (1/((1 + I1(w1, Delta_eq, eta, beta))*(1 + I1(w2, Delta_eq, eta, beta))))*((beta**2)*(Q(w1, w2, eta)/((2*np.pi*w1 + 1j*2*np.pi*eta)**2 - 4*(k**2 + 0.5*(beta**2)*((2*np.pi)**2)*Delta_eq)) + \
        Q(w2, w1, eta)/((2*np.pi*w2 + 1j*2*np.pi*eta)**2 - 4*(k**2 + 0.5*(beta**2)*((2*np.pi)**2)*Delta_eq))) - (1/(((2*np.pi)**2)*Delta_eq))*(1/(1 + I1(w1 + w2, Delta_eq, eta, beta))) * \
        (I2(w1 + w2, w1, Delta_eq, eta, beta)*Q(w1, w2, eta) + I2(w1 + w2, w2, Delta_eq, eta, beta)*Q(w2, w1, eta) + I1(w1, Delta_eq, eta, beta) + I1(w2, Delta_eq, eta, beta) + I1(w1, Delta_eq, eta, beta)*I1(w2, Delta_eq, eta, beta)))

chi2_vectorized = np.vectorize(chi2)

# Nonlinear response function (symmetrized, summed over k)
def chi2_sum(w1, w2, Delta_eq, eta, beta):
    return (1/(1 + I1(w1, Delta_eq, eta, beta)))*(1/(1 + I1(w2, Delta_eq, eta, beta)))*(1/(1 + I1(w1 + w2, Delta_eq, eta, beta))) * \
        (I2(w1 + w2, w1, Delta_eq, eta, beta)*Q(w1, w2, eta) + I2(w1 + w2, w2, Delta_eq, eta, beta)*Q(w2, w1, eta) - \
        I1(w1 + w2, Delta_eq, eta, beta)*(I1(w1, Delta_eq, eta, beta) + I1(w2, Delta_eq, eta, beta) + I1(w1, Delta_eq, eta, beta)*I1(w2, Delta_eq, eta, beta)))

chi2_sum_vectorized = np.vectorize(chi2_sum)

# SG Parameters

Delta_eq = 1.

betas = np.array([np.sqrt(np.pi), np.sqrt(np.pi/2.)])

peakval = -1

vmin = 0
vmax = 1

for beta in betas:

    ref_freq = np.sqrt(2*(beta**2)*Delta_eq)

    eta = 0.01*ref_freq

    ti = time.time()

    # Frequency window and grid

    window = 2*ref_freq

    N_grid = 1000

    w_grid = np.linspace(-window, window, 2*N_grid)

    w1_grid, w2_grid, = np.meshgrid(w_grid, w_grid)

    # Nonlinear response function in frequency domain

    mat = chi2_sum_vectorized(w1 = w1_grid, w2 = w2_grid, Delta_eq = Delta_eq, eta = eta, beta = beta)

    # Fourier transforming, now in time domain

    mat_time = np.fft.fftn(mat)

    mat_time = mat_time/np.abs(np.max(mat_time))

    tf = time.time()

    # Picking entries yielding the nonlinear response in time

    mat_t_tau = np.zeros((N_grid, N_grid), dtype = 'complex')

    for i in range(N_grid - 1):
        for j in range(N_grid - 1):
            mat_t_tau[i, j] = mat_time[i + j, j]

    # Fourier transforming back to frequency

    # map2d=np.fft.fft2(mat_t_tau)
    map2d = np.fft.fftshift(np.fft.fft2(mat_t_tau), 1)

    print(tf - ti)

    # Plotting

    w_plot = np.linspace(-window, window, N_grid)/ref_freq

    # This plot contains both the full map and the zoom

    fig = plt.figure(figsize = (11, 10))
    gs = GridSpec(nrows = 2, ncols = 2, wspace = 0.2, hspace = 0.2)

    ax0 = fig.add_subplot(gs[:,0])

    pcm = ax0.contourf(w_plot[int(N_grid/2) :], w_plot, np.log(np.abs(map2d)[:, int(N_grid/2) :]), cmap = cmap_2Dmaps, alpha = 0.75, levels = 8)
    ax0.contour(w_plot[int(N_grid/2) :], w_plot, np.log(np.abs(map2d)[:, int(N_grid/2) :]), cmap = cmap_2Dmaps, alpha = 0.75, levels = 8)

    # Zoom

    ax1 = fig.add_subplot(gs[0,1])

    center_tau = -0.985
    center_t = -0.985
    window_zoom = 0.15

    wtau_min = center_tau-window_zoom
    wtau_max = center_tau+window_zoom

    wt_min = -center_t-window_zoom
    wt_max = -center_t+window_zoom

    index_wtau_min = np.argmin(np.abs(w_plot - wtau_min))
    index_wtau_max = np.argmin(np.abs(w_plot - wtau_max))
    index_wt_min = np.argmin(np.abs(w_plot - wt_min))
    index_wt_max = np.argmin(np.abs(w_plot - wt_max))

    pcm = ax1.contourf(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],(np.abs(map2d)[index_wtau_min:index_wtau_max,index_wt_min:index_wt_max]),cmap=cmap_2Dmaps,alpha=0.75,levels=8)
    ax1.contour(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],(np.abs(map2d)[index_wtau_min:index_wtau_max,index_wt_min:index_wt_max]),cmap=cmap_2Dmaps,alpha=0.75,levels=8)

    # ax0.set_xticks([0,1])
    # ax0.set_yticks([-1,0,1])

    fig.savefig(os.path.join(wd, "rephasing_eta={:.3f}*ref_Delta={:.2f}_beta2={:.2f}.pdf".format(eta/ref_freq, Delta_eq, beta**2)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close(fig)

    # This is the zoom plot we actually care about

    if peakval == -1:
        normalized_data = np.abs(map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max]/np.max(np.abs(map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max])
        peakval = np.max(np.abs(map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max])
    else:
        normalized_data = np.abs(map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max]/peakval

    fig1, ax1 = plt.subplots(nrows = 1, ncols = 1)

    levels = np.linspace(vmin, vmax, 8)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax1.contourf(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,norm=norm,cmap=cmap_2Dmaps,alpha=0.75,levels=levels)
    ax1.contour(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,norm=norm,cmap=cmap_2Dmaps,alpha=0.75,levels=levels)

    cbar = fig1.colorbar(pcm, ax=ax1)
    cbar.set_label(r"$ \left| \sum_k \mathcal{{D}}_{{k,\rm{{nl}}}}^{{\varphi}} \right| \; \rm{{(normalized)}} $")

    tick_positions = np.linspace(vmin, vmax, 3)
    cbar.set_ticks(tick_positions)

    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax1.set_xlabel(r"$ \omega_2 / ( 2 M_{{B_1}} ) $")
    ax1.set_ylabel(r"$ \omega_1 / ( 2 M_{{B_1}} ) $")

    # ax1.set_xticks([])
    # ax1.set_yticks([])

    fig1.savefig(os.path.join(wd, "zoom_rephasing_eta={:.3f}*ref_Delta={:.2f}_beta2={:.2f}.pdf".format(eta/ref_freq, Delta_eq, beta**2)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close(fig1)