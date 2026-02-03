#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:28:19 2025

LMC - 30 Dor fragments

Function to estimate physical properties such as 
mass from given flux peak in mJy - CONVERTS flux peak
into integrated Gaussian beam flux in mJy/pixel 
for a given FWHM (in pixels) in pixel

Built CMF and statistics
@author: alessio
"""


from astropy.io import ascii
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.table import Table, QTable
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
import math
import sys


########################### Parameters ###########################
# Input files
clump = ['ALL']
nsigma = '5'

Hyper_py = 1    ## 1 if use Hyper_py results; 0 if use Hyper IDL results


# --- Input/output files --- #

if Hyper_py == 0: 
    dir_in = "/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Hyper/photometry/5sigma/"           # IDL version
    file_cores_base = dir_in + 'photometry_sources_' 

    # Output files
    dir_out = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Hyper-py/'
    dir_ps_out = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Hyper-py/' + 'ps_files/CDF/'
else:
    dir_in = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Hyper-py/output_final_review_01_2026/'
    file_cores_base = dir_in + 'params/Table_3_4_Clumps_Phys_props' 
    # Output files
    dir_out = dir_in
    dir_ps_out = dir_out + 'ps_files/CDF/'


file_params_out_base = dir_out + 'params/CDF/Table_CDF_out.txt'
file_params_out = file_params_out_base
# for cl in clump:
#     file_params_out += f'_{cl}'

# if Hyper_py == 0: 
#     file_params_out += '_IDL.txt'
# else: 
#     file_params_out += '_Python.txt'

file_out_ipac = dir_out + 'params/CDF/KS_realizations_params.txt'

# Plot file
if Hyper_py == 0: 
    file_out_CDF = dir_ps_out + 'CDF_fit_IDL.png'
    file_out_CDF_histo = dir_ps_out + 'CDF_histogram_IDL.png'
else:
    file_out_CDF = dir_ps_out + 'CDF_fit.png'
    file_out_CDF_histo = dir_ps_out + 'CDF_histogram.png'


# Constants
distance = 50.0 * u.kpc  # kpc
k_0_ref = 1.16 * u.g**-1 * u.cm**2 #1.16 * u.g**-1 * u.cm**2  # cm^2 g^-1 (from Brunetti+19 for LMC)
gas_dust_ratio = 500  # from Brunetti+19 for LMC
band_lambda = 1.3 * u.mm
fixed_temp = 36.7 * u.K  # Fixed temperature (from Brunetti+19)
temp_range_min, temp_range_max = 20.0, 80.0  # Min and max temperatures for Monte Carlo
beta = 2.0

c = const.c
k = const.k_B
h = const.h

# Constants for mass estimation
freq = c * (1. / band_lambda)  # Hz
freq = freq.to(u.Hz)
freq_0 = 2.3e+11 * u.Hz  # Hz
k_0 = k_0_ref / gas_dust_ratio * u.Hz  # cm^2 g^-1 * Hz


# Starting indices and parameters for fitting
seed = 1           # starting seed (each reference run is equal to the seed number)
min_start_index = 30  # Minimum starting index
min_points_for_fit = 30  # Minimum points for fit
n_realizations = 5000  # Number of Monte Carlo realizations

mass_completeness_limit = 5.0 #7.7  # M_sun - completeness limit

refer_slope = -1.65     # reference slope obtained from 36.7 K
schenider_18 = -0.90    # Schneider+18 30Dor reference slope
salpeter = -1.35     # Salpeter reference slope
start_mass = 1      # start mass value to plot salpeter slope

min_perc = 5       # min percentile for slope distribution
max_perc = 95       # max percentile for slope distribution


########################### Parameters ###########################



# Initialize arrays
my_clump, tot_flux, ra, dec, fwhm_avg = [], [], [], [], []

########################### Process Data ###########################

# Read and process flux data for each clump
for cl in clump:
    file_cores = file_cores_base + '.txt'
    data_cores = ascii.read(file_cores)

    temp_flux = data_cores['FLUX_INT']
    temp_ra = data_cores['RA']
    temp_dec = data_cores['DEC']
    
    temp_fwhm_1 = data_cores['FWHM_1']
    temp_fwhm_2 = data_cores['FWHM_2']
    temp_fwhm_avg = np.sqrt(temp_fwhm_1 * temp_fwhm_2)

    # Append data
    my_clump.extend([cl] * len(temp_flux))
    tot_flux.extend(temp_flux)
    ra.extend(temp_ra)
    dec.extend(temp_dec)
    fwhm_avg.extend(temp_fwhm_avg)

# Assign units to quantities
tot_flux = np.array(tot_flux) * u.mJy
ra = np.array(ra) * u.deg
dec = np.array(dec) * u.deg
fwhm_avg = np.array(fwhm_avg) * u.arcsec

# Estimate radius from FWHMs
distance_pc = distance.to(u.pc)  # Convert to pc
fwhm_avg_rad = fwhm_avg * (np.pi / 648000)
radius_pc = fwhm_avg_rad * distance_pc / u.arcsec  # Convert to pc
radius_cm = radius_pc.to(u.cm)
radius_AU = radius_pc.to(u.AU)


# Calcualte mass for each clump using the fixed temperature (36.7 K)
fixed_mass_g = []

for i in range(len(tot_flux)):
    # Use the fixed temperature for all clumps
    T_fixed = fixed_temp
    
    # Recalculate k_nu with the fixed temperature
    k_nu_fixed = (k_0_ref / gas_dust_ratio) * (freq / freq_0) ** beta
    
    # Recalculate the greybody spectrum (bb) with the fixed temperature
    bb_fixed = 2 * h * freq**3 / c**2 * (1 / (np.exp(h * freq / (k * T_fixed)) - 1))
    bb_Jy_fixed = bb_fixed.to(u.Jy)
    
    # Recalculate mass using the fixed temperature
    mass_g_fixed = (tot_flux[i].to(u.Jy) * distance.to(u.cm)**2) / (k_nu_fixed * bb_Jy_fixed)
    
    # Convert mass from grams to solar masses and append as a Quantity
    fixed_mass_g.append(mass_g_fixed.to(u.solMass))

# Now fixed_mass_g is a list of astropy Quantity objects with solar mass units
# You can then work with them as Quantity objects

# Optionally, if you need the raw values for numpy operations:
fixed_mass_values = np.array([mass.value for mass in fixed_mass_g])  # Convert to numpy array of dimensionless values

# Convert the masses to solar masses
fixed_source_solar_mass = fixed_mass_values * u.solMass

# Sort the fixed masses and compute the CDF
sorted_fixed_mass = np.sort(fixed_source_solar_mass)

# Compute the cumulative distribution function (CDF) for the fixed temperatures
fixed_cdf = np.arange(1, len(sorted_fixed_mass) + 1) / len(sorted_fixed_mass)

# Compute the complementary cumulative distribution function (CCDF) for the fixed temperatures
fixed_ccdf = (len(sorted_fixed_mass) - np.arange(0, len(sorted_fixed_mass))) / len(sorted_fixed_mass)




# Define storage arrays for all realizations
all_random_mass_values = []  # Stores raw mass values for each realization
all_sorted_random_mass = []  # Stores sorted mass values for each realization
all_random_ccdfs = []        # Stores CCDFs for each realization

# Run n_realization realizations of random temperatures
rng = np.random.default_rng(seed=seed)
for _ in range(n_realizations):
    # Generate random temperatures
    random_temps = rng.uniform(temp_range_min, temp_range_max, size=len(tot_flux)) * u.K

    # Recalculate masses for each realization
    random_mass_g = []
    for i in range(len(tot_flux)):
        T_random = random_temps[i]
        k_nu_random = (k_0_ref / gas_dust_ratio) * (freq / freq_0) ** beta
        bb_random = 2 * h * freq**3 / c**2 * (1 / (np.exp(h * freq / (k * T_random)) - 1))
        bb_Jy_random = bb_random.to(u.Jy)
        mass_g_random = (tot_flux[i].to(u.Jy) * distance.to(u.cm)**2) / (k_nu_random * bb_Jy_random)
        random_mass_g.append(mass_g_random.to(u.solMass))

    # Convert mass list to numpy array
    random_mass_values = np.array([mass.value for mass in random_mass_g])  # Raw values
    all_random_mass_values.append(random_mass_values)

    # Sort the masses and calculate CCDF
    sorted_mass = np.sort(random_mass_values)  # Sort the masses
    ccdf = (len(sorted_mass) - np.arange(0, len(sorted_mass))) / len(sorted_mass)  # CCDF
    all_sorted_random_mass.append(sorted_mass)
    all_random_ccdfs.append(ccdf)

# Convert lists to numpy arrays
all_sorted_random_mass = np.array(all_sorted_random_mass)  # Shape (100, num_clumps)
all_random_ccdfs = np.array(all_random_ccdfs)              # Shape (100, num_clumps)


################## Function Definitions ##################

# Define the power-law function for fitting
def power_law(x, a, b):
    return a * x**b

# Function to calculate the Kolmogorov-Smirnov (KS) distance
def ks_distance(data_mass, data_ccdf, min_start_index=0):
    popt, _ = curve_fit(power_law, data_mass[min_start_index:], data_ccdf[min_start_index:], maxfev=10000)
    expected_ccdf = power_law(data_mass[min_start_index:], *popt)
    ks_statistic, _ = ks_2samp(data_ccdf[min_start_index:], expected_ccdf)
    return ks_statistic

# Function to calculate the power-law fit
def fit_power_law(masses, ccdf, start_index, min_points_for_fit):
    return curve_fit(power_law, masses[start_index:start_index+min_points_for_fit], ccdf[start_index:start_index+min_points_for_fit])

# Function to calculate the KS distance allowing for varying start and end points
def ks_distance_varying_range(data_mass, data_ccdf, start_index, end_index):
    popt, _ = curve_fit(power_law, data_mass[start_index:end_index], data_ccdf[start_index:end_index], maxfev=10000)
    expected_ccdf = power_law(data_mass[start_index:end_index], *popt)
    ks_statistic, _ = ks_2samp(data_ccdf[start_index:end_index], expected_ccdf)
    return ks_statistic

# Function to find the optimal fitting window by varying both start and end indices
def find_optimal_fitting_window(masses, ccdf, min_start_index, min_points_for_fit):
    ks_results = []

    # Iterate over all valid start points
    for start in range(min_start_index, len(masses) - min_points_for_fit):
        # Iterate over all valid end points
        for end in range(start + min_points_for_fit, len(masses) + 1):
            ks_stat = ks_distance_varying_range(masses, ccdf, start, end)
            ks_results.append((start, end, ks_stat))

    # Find the window (start, end) with the minimum KS distance
    optimal_window = min(ks_results, key=lambda x: x[2])  # Minimize KS distance
    optimal_start, optimal_end, min_ks_distance = optimal_window
    return optimal_start, optimal_end, min_ks_distance




def plot_ccdfs_and_fits():
    """Plot all CCDFs, fits, and identify slope extremes."""
      
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set axis labels and title with bold font and size 12
    ax.set_xlabel(r'Solar Mass (M$_{\odot}$)', fontsize=12)
    ax.set_ylabel('Inverse Cumulative Density', fontsize=12)
    
    # Make tick labels bold and size 12
    ax.tick_params(axis='both', which='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontweight('bold')
        
        # Make all spines (axes) bold
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    
  
    # Plot all CCDFs for the random temperature realizations
    for i in range(n_realizations):
        plt.plot(
            all_sorted_random_mass[i], all_random_ccdfs[i],
            color='green', alpha=0.01, label='CCDFs with 20$\leq$ T $\leq80$ K' if i == 0 else None
        )

    
    
    # Get the mass and CCDF for the reference T=36.7 K realization
    mass = sorted_fixed_mass
    ccdf = fixed_ccdf
    color_ref_value = 'blue'
    
    # Find the optimal fitting window (start and end indices)
    start_index, end_index, _ = find_optimal_fitting_window(mass, ccdf, min_start_index, min_points_for_fit)

    # Filter the mass and CCDF for fitting using the optimal window
    filtered_mass = mass[start_index:end_index]
    filtered_ccdf = ccdf[start_index:end_index]

    # Fit the power-law to the filtered data
    popt, _ = curve_fit(power_law, filtered_mass, filtered_ccdf)

    refer_slope = popt[1]
    

    # Plot the CCDF for the realization
    plt.plot(
        mass, ccdf, color=color_ref_value, linestyle='-', alpha=0.7
    )
        # mass, ccdf, color=color, linestyle='-', alpha=0.7,
        # label=f'{label}: $x^{{{exponents[exponent_idx]:.2f}}}$'

    
    # Plot the fitted power-law across the entire mass range
    plt.plot(
        mass, power_law(mass, *popt), color=color_ref_value, linestyle='--', 
        alpha=1,
        label=f'Ref. (T=36.7 K) Fit: $x^{{{refer_slope:.2f}}}$'
    )


     # Overplot the points used for the power-law fit (with filled circles)
    plt.scatter(
        filtered_mass, filtered_ccdf, color=color_ref_value, edgecolor='black', s=50, zorder=5,
    )    
    #     label=f'{label} Data Points'  # Add label for the points
    # )

    # Overplot the unused points with a lighter color (alpha transparency)
    unused_mass = np.concatenate([mass[:start_index], mass[end_index:]])
    unused_ccdf = np.concatenate([ccdf[:start_index], ccdf[end_index:]])

    # Plot unused points with lighter color and smaller size
    plt.scatter(
        unused_mass, unused_ccdf, color=color_ref_value, edgecolor='grey', 
        s=30, alpha=0.3, zorder=4,
    )
    #     label=f'{label} Unused Points'  # Label for unused points
    # )    

    



    # Identify and plot the minimum and maximum slope realizations
    plot_extreme_slope_realization(min_exponent_idx, 'purple', 'Min Slope', min_start_index, min_points_for_fit)
    plot_extreme_slope_realization(max_exponent_idx, 'orange', 'Max Slope', min_start_index, min_points_for_fit)

    # Overplot the reference power-law with an exponent of salpeter
    overplot_reference_power_law(
        exponent=salpeter, color='green', linestyle='--', 
        label=f'Salpeter: $x^{{{salpeter:.2f}}}$'
    )

    # Set logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

        

    # Adjust y-axis limits based on the CCDF data
    min_ccdf_value = np.min(fixed_ccdf[min_start_index:])
    max_ccdf_value = np.max(fixed_ccdf[min_start_index:])
    plt.ylim(bottom=min_ccdf_value * 0.8, top=1.2)  # Set limits slightly below and above the observed data
    # plt.xlim(1,1.e3)  # Set limits slightly below and above the observed data

    # Add legend and show the plot
    # plt.legend()
    plt.legend(loc='lower left', fontsize=8)   # bbox_to_anchor=(1.2, 1), 
    plt.savefig(file_out_CDF, dpi=300, bbox_inches='tight')
    plt.show()    
    




def overplot_reference_power_law(exponent=salpeter, color='green', linestyle='--', label=None):
    """
    Overplots a reference power-law on the CCDF plot, starting at the minimum value of the mass.

    Parameters:
    -----------
    exponent : float
        The exponent of the reference power-law.
    color : str
        The color of the power-law line.
    linestyle : str
        The linestyle for the power-law line.
    label : str
        Label for the plot legend. If None, it will default to 'Reference Power-Law'.
    """
    # Generate mass values for the reference line starting from the minimum mass
    reference_mass = np.logspace(np.log10(start_mass), 
                                 np.log10(max(sorted_fixed_mass.value)), 100)
    
    # Compute the reference CCDF values using the power-law formula
    reference_ccdf = reference_mass**exponent

    # Normalize the reference CCDF to start at the same value as the CCDF at min mass
    normalization_factor = fixed_ccdf[0] / reference_ccdf[0]
    reference_ccdf *= normalization_factor

    # Plot the reference power-law
    
    plt.plot(reference_mass, reference_ccdf, color=color, linestyle=linestyle, 
             alpha=0.7, label=label or f'Reference: $x^{{{exponent:.2f}}}$')    


    # Add vertical line for "Mass Completeness Limit"
    plt.axvline(
    x=mass_completeness_limit, color='lightblue', linestyle='-', 
    linewidth=1.5, label='Mass Compl. Limit (T=36.7 K)'
    )
    
    


# Function to plot the histogram of exponents
def plot_exponent_histogram(exponents, exponent_fixed, refer_slope):
    """Plot the histogram of exponents and highlight the fixed temperature exponent, mean, median, and 10%-90% range."""
    
    # Calculate the mean, median, and the 10th and 90th percentiles of the exponent distribution
    mean_exponent = np.mean(exponents)
    median_exponent = np.median(exponents)
    percentile_min = np.percentile(exponents, min_perc)
    percentile_max = np.percentile(exponents, max_perc)
    
 
    # Compute mean (mu) and standard deviation (sigma)
    mu = np.mean(exponents)
    sigma = np.std(exponents, ddof=1)  # Use ddof=1 for an unbiased estimator

    print(f"Mean (μ): {mu:.4f}")
    print(f"Standard deviation (σ): {sigma:.4f}") 
 
    # # Compute statistics   
    # percentile_rank = stats.percentileofscore(exponents, value_to_check)

    # # Compute empirical p-value
    # p_value = min(percentile_rank, 100 - percentile_rank) / 100

    # print(f"Salpeter Percentile Rank: {percentile_rank:.2f}%")
    # print(f"Salpeter Empirical p-value: {p_value:.4f}")
 
    
   
    # Sort the data
    sorted_slopes = np.sort(exponents)
    
    # Define target value: Salpeter
    value_to_check = salpeter
    
    # Compute the empirical CDF
    ecdf = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
    
    # Find the index of the value of interest (-1.35)
    value_to_check = -1.35
    index = np.searchsorted(sorted_slopes, value_to_check)
    
    # Compute the empirical p-value (percentile)
    empirical_p_value = ecdf[index]  # This is the probability that the value is <= -1.35
    
    # You can also compute a two-tailed p-value for extreme values on both sides
    two_tailed_p_value = min(empirical_p_value, 1 - empirical_p_value) * 2
    
    print(f"Salpeter Empirical p-value for value {value_to_check}: {empirical_p_value:.4f}")
    print(f"Salpeter Two-tailed empirical p-value: {two_tailed_p_value:.4f}")

 
    
    # Define target value: Schneider+18
    value_to_check = schenider_18
    
    # Compute the empirical CDF
    ecdf = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
    
    # Find the index of the value of interest (-1.35)
    index = np.searchsorted(sorted_slopes, value_to_check)
    
    # Compute the empirical p-value (percentile)
    empirical_p_value = ecdf[index]  # This is the probability that the value is <= -0.90
    
    # You can also compute a two-tailed p-value for extreme values on both sides
    two_tailed_p_value = min(empirical_p_value, 1 - empirical_p_value) * 2
    
    print(f"Schneider+18 Empirical p-value for value {value_to_check}: {empirical_p_value:.4f}")
    print(f"Schneider+18 Two-tailed empirical p-value: {two_tailed_p_value:.4f}")
  
    
 
    # Define target value: Reference temperature T=36.7 K
    value_to_check = refer_slope
    
    # Compute the empirical CDF
    ecdf = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
    
    # Find the index of the value of interest (-1.35)
    index = np.searchsorted(sorted_slopes, value_to_check)
    
    # Compute the empirical p-value (percentile)
    empirical_p_value = ecdf[index]  # This is the probability that the value is <= -1.35
    
    # You can also compute a two-tailed p-value for extreme values on both sides
    two_tailed_p_value = min(empirical_p_value, 1 - empirical_p_value) * 2
    
    print(f"{refer_slope} T=36.7 K Empirical p-value for value {value_to_check}: {empirical_p_value:.4f}")
    print(f"{refer_slope} T=36.7 K Two-tailed empirical p-value: {two_tailed_p_value:.4f}")
     
    
    
    
    # Plot histogram of the exponents
    
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set axis labels and title with bold font and size 12
    ax.set_xlabel(r'Slope (exponent)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Make tick labels bold and size 12
    ax.tick_params(axis='both', which='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontweight('bold')
        
        # Make all spines (axes) bold
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
 
    
    # plt.figure(figsize=(8, 6))
    plt.hist(exponents, bins=30, color='c', alpha=0.7, edgecolor='k', label=f'All Distr. ({temp_range_min:.0f}$\leq T\leq${temp_range_max:.0f} K)')

    # Add vertical dashed line for the fixed temperature exponent
    plt.axvline(x=exponent_fixed, color='b', linestyle='--', linewidth=2, label=f'Fixed T=36.7 K: {refer_slope:.2f}')

    # Add vertical dashed line for the Salpeter exponent
    plt.axvline(x=salpeter, color='g', linestyle='--', linewidth=2, label=f'Salpeter: {salpeter:.2f}')

    # Add vertical dashed line for the Schneider+18 IMF exponent
    plt.axvline(x=schenider_18, color='black', linestyle='--', linewidth=2, label=f'Schneider+18: {schenider_18:.2f}')

    # Add vertical dashed line for the mean of the exponent distribution
    plt.axvline(x=mean_exponent, color='r', linestyle='-.', linewidth=2, label=f'Mean: {mean_exponent:.2f}')

    # Add vertical dashed line for the median of the exponent distribution
    plt.axvline(x=median_exponent, color='m', linestyle=':', linewidth=2, label=f'Median: {median_exponent:.2f}')

    # Add shaded area for the 10%-90% distribution of the slopes
    plt.fill_betweenx(
        [0, plt.gca().get_ylim()[1]],  # Y-range from 0 to the top of the histogram
        percentile_min, percentile_max,  # X-range between 10th and 90th percentiles
        color='y', alpha=0.3, label=f'{min_perc}%-{max_perc}% Range: {percentile_min:.2f} : {percentile_max:.2f}'
    )

    # Add labels, title, and legend
    # plt.xlabel('Slope (Exponent)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Slopes from Random Temperature Realizations')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig(file_out_CDF_histo, dpi=300, bbox_inches='tight')
    plt.show()
    



def plot_extreme_slope_realization(exponent_idx, color, label, min_start_index, min_points_for_fit):
    """
    Plot a specific realization (either min or max slope) with power-law fit based on an optimal fitting window.

    Parameters:
        exponent_idx (int): Index of the exponent realization to plot.
        color (str): Color for the plot.
        label (str): Label for the plot.
        min_start_index (int): Minimum starting index for fitting.
        min_points_for_fit (int): Minimum number of points required for fitting.
    """
    # Get the mass and CCDF for the current realization
    mass = all_sorted_random_mass[exponent_idx]
    ccdf = all_random_ccdfs[exponent_idx]
    
    # Find the optimal fitting window (start and end indices)
    start_index, end_index, _ = find_optimal_fitting_window(mass, ccdf, min_start_index, min_points_for_fit)

    # Filter the mass and CCDF for fitting using the optimal window
    filtered_mass = mass[start_index:end_index]
    filtered_ccdf = ccdf[start_index:end_index]

    # Fit the power-law to the filtered data
    popt, _ = curve_fit(power_law, filtered_mass, filtered_ccdf)

    # Plot the CCDF for the realization
    plt.plot(
        mass, ccdf, color=color, linestyle='-', alpha=0.7
    )
        # mass, ccdf, color=color, linestyle='-', alpha=0.7,
        # label=f'{label}: $x^{{{exponents[exponent_idx]:.2f}}}$'

    
    # Plot the fitted power-law across the entire mass range
    plt.plot(
        mass, power_law(mass, *popt), color=color, linestyle='--', 
        alpha=1,
        label=f'{label} Fit: $x^{{{popt[1]:.2f}}}$'
    )


     # Overplot the points used for the power-law fit (with filled circles)
    plt.scatter(
        filtered_mass, filtered_ccdf, color=color, edgecolor='black', s=50, zorder=5,
    )    
    #     label=f'{label} Data Points'  # Add label for the points
    # )

    # Overplot the unused points with a lighter color (alpha transparency)
    unused_mass = np.concatenate([mass[:start_index], mass[end_index:]])
    unused_ccdf = np.concatenate([ccdf[:start_index], ccdf[end_index:]])

    # Plot unused points with lighter color and smaller size
    plt.scatter(
        unused_mass, unused_ccdf, color=color, edgecolor='grey', 
        s=30, alpha=0.3, zorder=4,
    )
    #     label=f'{label} Unused Points'  # Label for unused points
    # )    
    


    
################## Calculations ##################

## Initialize lists to store results
exponents = []
optimal_windows = []  # Store optimal (start, end) indices
min_ks_distances = []

# Iterate over all realizations
for i in range(n_realizations):
    masses = all_sorted_random_mass[i]
    ccdf = all_random_ccdfs[i]

    # Find the optimal fitting window for this realization
    optimal_start, optimal_end, min_ks_distance = find_optimal_fitting_window(
        masses, ccdf, min_start_index, min_points_for_fit
    )

    # Store the results
    optimal_windows.append((optimal_start, optimal_end))
    min_ks_distances.append(min_ks_distance)

    # Perform the power-law fit for the optimal window
    filtered_mass = masses[optimal_start:optimal_end]
    filtered_ccdf = ccdf[optimal_start:optimal_end]
    popt, _ = curve_fit(power_law, filtered_mass, filtered_ccdf)
    _, exponent = popt
    exponents.append(exponent)

# Find the realizations with minimum and maximum exponents
min_exponent_idx = np.argmin(exponents)
max_exponent_idx = np.argmax(exponents)

# To ensure the final 100% is printed
sys.stdout.write("\rProgress: 100%\n")
sys.stdout.flush()


################## Save Results ##################

# Save the optimal fitting windows, minimum KS distances, and power-law exponents
with open(file_out_ipac, 'w') as f:
    # Write the header
    f.write("| Realization | Start Index | End Index | Minimum KS Distance | Power-Law Exponent |\n")
    f.write("|-------------|-------------|-----------|----------------------|---------------------|\n")

    # Write each realization's results along with the power-law exponent
    for i, (start, end, ks_distance) in enumerate(zip(*zip(*optimal_windows), min_ks_distances)):
        # Get the mass and CCDF for the current realization
        mass = all_sorted_random_mass[i]
        ccdf = all_random_ccdfs[i]
        
        # Find the optimal fitting window (start and end indices)
        filtered_mass = mass[start:end]
        filtered_ccdf = ccdf[start:end]
        
        # Fit the power-law to the filtered data
        popt, _ = curve_fit(power_law, filtered_mass, filtered_ccdf)
        
        # Write the results, including the exponent (second parameter of popt)
        f.write(f"{i+1:>13} {start:>13} {end:>11} {ks_distance:>22.5f} {popt[1]:>20.5f}\n")

print(f"KS Distances and Power-Law Exponents saved to {file_out_ipac}")

################## Plotting ##################

# Perform the power-law fit for the fixed temperature CCDF
popt_fixed, _ = fit_power_law(sorted_fixed_mass, fixed_ccdf, 31, 40)
a_fixed, exponent_fixed = popt_fixed  # Extract the coefficient and exponent

plt.clf()
# Plot all CCDFs, fits, and slope extremes
plot_ccdfs_and_fits()

plt.clf()
# Plot the histogram of the exponents
plot_exponent_histogram(exponents, exponent_fixed, refer_slope)
