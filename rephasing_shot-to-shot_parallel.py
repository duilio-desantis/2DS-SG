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
from concurrent.futures import ProcessPoolExecutor
import matplotlib.ticker as ticker

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

def compute_and_accumulate(Delta, gw_element, w1_grid, w2_grid, eta, beta, N_grid):
    print(f"Process {os.getpid()} working on Delta={Delta}")

    # Nonlinear response function in frequency domain
    mat = chi2_sum_vectorized(w1=w1_grid, w2=w2_grid, Delta_eq=Delta, eta=eta, beta=beta)

    # Fourier transform to time domain
    mat_time = np.fft.fftn(mat)
    mat_time = mat_time / np.abs(np.max(mat_time))  # Normalize

    # Picking entries for the nonlinear response in time
    mat_t_tau = np.zeros((N_grid, N_grid), dtype='complex')
    for i in range(N_grid - 1):
        for j in range(N_grid - 1):
            mat_t_tau[i, j] = mat_time[i + j, j]

    # Return the weighted matrix
    return mat_t_tau * gw_element

def compute_avg_mat_t_tau(Deltas, w1_grid, w2_grid, eta, beta, gw, N_grid):
    
    avg_mat_t_tau = np.zeros((N_grid, N_grid), dtype='complex')

    with ProcessPoolExecutor() as executor:
        
        for mat_t_tau in executor.map(compute_and_accumulate, Deltas, gw, [w1_grid] * len(Deltas), [w2_grid] * len(Deltas), \
            [eta] * len(Deltas), [beta] * len(Deltas), [N_grid] * len(Deltas),): avg_mat_t_tau += mat_t_tau

    return avg_mat_t_tau

def main():

    # SG Parameters

    Delta_eq = 1.
    #Average value

    beta = np.sqrt(2.*np.pi)
    #Fixed

    ref_freq = np.sqrt(2*(beta**2)*Delta_eq)
    #Reference to average

    eta = 0.01*ref_freq

    # Frequency window and grid
    
    window = 2*ref_freq

    N_grid = 400

    w_grid = np.linspace(-window, window, 2*N_grid)

    w1_grid, w2_grid, = np.meshgrid(w_grid, w_grid)

    # 2B_1 frequencies (gaps) shot-to-shot to explore
    sd = 0.1*ref_freq
    freqs = np.linspace(ref_freq - 4*sd, ref_freq + 4*sd, 100)

    # Gaussian weight
    gw = np.exp(-((freqs - ref_freq)**2)/(2*(sd**2)))

    plt.plot(freqs/ref_freq, gw, 'C0', linewidth = 2)
    plt.axvline(1., c = 'C1', linestyle = '--', label = r"$ \omega_{{\rm{{avg}}}} / \omega_{{\rm{{ref}}}} = 1 $")
    plt.xlabel(r"$ \omega / \omega_{{\rm{{ref}}}} $")
    plt.legend()
    plt.savefig(os.path.join(wd, "freqs_scaled_avg={:.2f}*ref_sd={:.2f}*ref_Delta-eq={:.2f}.pdf".format(1., sd/ref_freq, Delta_eq)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close()

    Deltas = freqs**2/(2.*beta**2)

    # Computing average matrix
    avg_mat_t_tau = compute_avg_mat_t_tau(Deltas, w1_grid, w2_grid, eta, beta, gw, N_grid)

    # Fourier transforming back to frequency
    avg_map2d = np.fft.fftshift(np.fft.fft2(avg_mat_t_tau), 1)

    # Plotting

    w_plot = np.linspace(-window, window, N_grid)/ref_freq

    # This plot contains both the full map and the zoom

    fig = plt.figure(figsize = (11, 10))
    gs = GridSpec(nrows = 2, ncols = 2, wspace = 0.2, hspace = 0.2)

    ax0 = fig.add_subplot(gs[:,0])

    pcm = ax0.contourf(w_plot[int(N_grid/2) :], w_plot, np.log(np.abs(avg_map2d)[:, int(N_grid/2) :]), cmap = cmap_2Dmaps, alpha = 0.75, levels = 8)
    ax0.contour(w_plot[int(N_grid/2) :], w_plot, np.log(np.abs(avg_map2d)[:, int(N_grid/2) :]), cmap = cmap_2Dmaps, alpha = 0.75, levels = 8)

    # Zoom

    ax1 = fig.add_subplot(gs[0,1])

    center_tau = -0.95
    center_t = -0.975
    window_zoom = 0.5

    wtau_min = center_tau-window_zoom
    wtau_max = center_tau+window_zoom

    wt_min = -center_t-window_zoom
    wt_max = -center_t+window_zoom

    index_wtau_min = np.argmin(np.abs(w_plot - wtau_min))
    index_wtau_max = np.argmin(np.abs(w_plot - wtau_max))
    index_wt_min = np.argmin(np.abs(w_plot - wt_min))
    index_wt_max = np.argmin(np.abs(w_plot - wt_max))

    pcm = ax1.contourf(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],(np.abs(avg_map2d)[index_wtau_min:index_wtau_max,index_wt_min:index_wt_max]),cmap=cmap_2Dmaps,alpha=0.75,levels=8)
    ax1.contour(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],(np.abs(avg_map2d)[index_wtau_min:index_wtau_max,index_wt_min:index_wt_max]),cmap=cmap_2Dmaps,alpha=0.75,levels=8)

    # ax0.set_xticks([0,1])
    # ax0.set_yticks([-1,0,1])

    fig.savefig(os.path.join(wd, "rephasing_eta={:.3f}*ref_Delta-eq={:.2f}_beta2={:.2f}.pdf".format(eta/ref_freq, Delta_eq, beta**2)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close(fig)

    # This is the rephasing zoom plot we actually care about

    normalized_data = np.abs(avg_map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max]/np.max(np.abs(avg_map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max])

    fig1, ax1 = plt.subplots(nrows = 1, ncols = 1)

    pcm = ax1.contourf(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,cmap=cmap_2Dmaps,alpha=0.75,levels=8)
    ax1.contour(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,cmap=cmap_2Dmaps,alpha=0.75,levels=8)

    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    cbar = fig1.colorbar(pcm, ax=ax1)
    cbar.set_label(r"$ \left| \sum_k \mathcal{{D}}_{{k,\rm{{nl}}}}^{{\varphi}} \right| \; \rm{{(normalized)}} $")

    tick_positions = np.linspace(0., 1., 3)
    cbar.set_ticks(tick_positions)

    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax1.set_xlabel(r"$ \omega_2 / ( 2 \overline{{M}}_{{B_1}} ) $")
    ax1.set_ylabel(r"$ \omega_1 / ( 2 \overline{{M}}_{{B_1}} ) $")

    # ax1.set_xticks([])
    # ax1.set_yticks([])

    fig1.savefig(os.path.join(wd, "zoom_rephasing_eta={:.3f}*ref_Delta-eq={:.2f}_beta2={:.2f}.pdf".format(eta/ref_freq, Delta_eq, beta**2)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close(fig1)

    # This is the non-rephasing zoom plot for comparison

    center_tau = 1.
    center_t = 0.95

    wtau_min = center_tau-window_zoom
    wtau_max = center_tau+window_zoom

    wt_min = center_t-window_zoom
    wt_max = center_t+window_zoom

    index_wtau_min = np.argmin(np.abs(w_plot - wtau_min))
    index_wtau_max = np.argmin(np.abs(w_plot - wtau_max))
    index_wt_min = np.argmin(np.abs(w_plot - wt_min))
    index_wt_max = np.argmin(np.abs(w_plot - wt_max))

    normalized_data = np.abs(avg_map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max]/np.max(np.abs(avg_map2d)[index_wtau_min:index_wtau_max, index_wt_min:index_wt_max])

    fig1, ax1 = plt.subplots(nrows = 1, ncols = 1)

    pcm = ax1.contourf(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,cmap=cmap_2Dmaps,alpha=0.75,levels=8)
    ax1.contour(w_plot[index_wt_min:index_wt_max],w_plot[index_wtau_min:index_wtau_max],normalized_data,cmap=cmap_2Dmaps,alpha=0.75,levels=8)

    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    cbar = fig1.colorbar(pcm, ax=ax1)
    cbar.set_label(r"$ \left| \sum_k \mathcal{{D}}_{{k,\rm{{nl}}}}^{{\varphi}} \right| \; \rm{{(normalized)}} $")

    tick_positions = np.linspace(0., 1., 3)
    cbar.set_ticks(tick_positions)

    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax1.set_xlabel(r"$ \omega_2 / ( 2 \overline{{M}}_{{B_1}} ) $")
    ax1.set_ylabel(r"$ \omega_1 / ( 2 \overline{{M}}_{{B_1}} ) $")

    # ax1.set_xticks([])
    # ax1.set_yticks([])

    fig1.savefig(os.path.join(wd, "zoom_non-rephasing_eta={:.3f}*ref_Delta-eq={:.2f}_beta2={:.2f}.pdf".format(eta/ref_freq, Delta_eq, beta**2)), format = "pdf", bbox_inches = 'tight', dpi = 500)
    plt.close(fig1)

if __name__ == "__main__":
    main()