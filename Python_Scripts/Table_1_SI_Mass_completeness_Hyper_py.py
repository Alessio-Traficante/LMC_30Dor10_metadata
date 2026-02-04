#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:49:36 2025

@author: alessio
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:12:01 2025

Mass_completeness_Hyper_vs_injected

@author: alessio
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from astropy.table import Table
from astropy import coordinates as coords
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u





# ################## Parameters ##################
num_sources = 500     # Tot sources
n_sigma = 4
dir_in_rms = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Final_data_images/Data_final_Table_01_2026/Supplementary_Information/Mass_Completeness/Reference_params/'
dir_in_Hyper = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Final_data_images/Data_final_Table_01_2026/Supplementary_Information/Mass_Completeness/'+str(n_sigma)+'sigma/Params/'



### params out ###
dir_params = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Final_data_images/Data_final_Table_01_2026/Supplementary_Information/Mass_Completeness/'+str(n_sigma)+'sigma/Params/Completeness_counts/'
table_source_count = dir_params + 'sources_counts_'+str(n_sigma)+'sigma.txt'


### files out ###
dir_ps_out = '/Users/alessio/Dropbox/Work/ALMA/ALMA_LMC/Final_data_images/Data_final_Table_01_2026/Supplementary_Information/Mass_Completeness/Images/'

tot_maps = 10
tolerance = 0.05 * u.arcsec

legend_properties = font_manager.FontProperties(size=12)
################## done !!! ##################


# Initialize lists to store completeness curves for each catalog
all_completeness = []
all_bin_centers = []



# Prepare to save values in an IPAC Table
data = Table(
    names=['Catalogue', 'Total_Sources', 'Common_Sources', 'Total_Sources_Hyper', 'False_Sources', 'False_Sources_Percentage'],
    dtype=['i4', 'i4', 'i4', 'i4', 'i4', 'f4']  # Integer and float types
)



# Loop over the 10 catalog pairs
for i in range(1, tot_maps+1):  # Assuming catalog files are named as 1.txt, 2.txt, ..., 10.txt
    # Step 1: Read the RMS and Hyper catalogs
    table_rms = Table.read(f'{dir_in_rms}table_'+str(num_sources)+'_Gaussians_'+str(i)+'.txt', format='ipac')
    table_hyper = Table.read(f'{dir_in_Hyper}hyper_output_map_'+str(num_sources)+'_Gaussians_'+str(i)+'.txt', format='ipac')


    # Step 2: Extract RA, DEC from both tables and add units
    ra_rms, dec_rms = table_rms['RA'], table_rms['DEC']
    ra_hyper, dec_hyper = table_hyper['RA'], table_hyper['DEC']
    table_rms['RA'] = ra_rms * u.deg
    table_rms['DEC'] = dec_rms * u.deg
    table_hyper['RA'] = ra_hyper * u.deg
    table_hyper['DEC'] = dec_hyper * u.deg

    # Step 3: Use Astropy coordinates for both catalogs
    coords_rms = coords.SkyCoord(ra=ra_rms, dec=dec_rms, unit='deg', frame='icrs')
    coords_hyper = coords.SkyCoord(ra=ra_hyper, dec=dec_hyper, unit='deg', frame='icrs')

    # Step 4: Compute pairwise separations dynamically
    if len(coords_rms) <= len(coords_hyper):
        separations = coords_rms[:, None].separation(coords_hyper[None, :])
        source_catalog = "RMS-to-Hyper"
    else:
        separations = coords_hyper[:, None].separation(coords_rms[None, :])
        source_catalog = "Hyper-to-RMS"

    # Convert separations to arcseconds
    separations_arcsec = separations.to(u.arcsec)

    # Find the minimum separation for each source
    min_separations = separations_arcsec.min(axis=1)
    min_indices = separations_arcsec.argmin(axis=1)

    # Step 5: Set a tolerance for matching
    matches = min_separations < tolerance
    
    
    ### Added a filter to keep only unique rms sources ###
    # Find the unique RMS matches, keeping the first occurrence
    _, unique_indices = np.unique(min_indices[matches], return_index=True)

    # Create a new mask that retains only the first occurrence of each match
    filtered_matches = np.zeros_like(matches, dtype=bool)
    filtered_matches[np.where(matches)[0][unique_indices]] = True
    
    # Use filtered_matches instead of matches in the if-else block
    matches = filtered_matches  # Overwrite the original matches
 

    # Extract the matched flux values and indices based on source catalog
    if source_catalog == "RMS-to-Hyper":
        flux_peak_rms = table_rms['Flux_Peak'][matches]
        flux_integrated_rms = table_rms['Flux_Integrated'][matches]
        flux_peak_hyper = table_hyper['FLUX_PEAK'][min_indices[matches]] * 1.e-3   # Jy -> mJy
        flux_integrated_hyper = table_hyper['FLUX'][min_indices[matches]] * 1.e-3   # Jy -> mJy

        # Save indices
        rms_indices_matched = np.where(matches)[0]         # Indices of matched RMS sources
        hyper_indices_matched = min_indices[matches]       # Corresponding Hyper source indices

    else:
        flux_peak_hyper = table_hyper['FLUX_PEAK'][matches] 
        flux_integrated_hyper = table_hyper['FLUX'][matches] * 1.e-3     # Jy -> mJy
        flux_peak_rms = table_rms['Flux_Peak'][min_indices[matches]]
        flux_integrated_rms = table_rms['Flux_Integrated'][min_indices[matches]]

        # Save indices
        hyper_indices_matched = np.where(matches)[0]       # Indices of matched Hyper sources
        rms_indices_matched = min_indices[matches]         # Corresponding RMS source indices    


    # count false identifications
    false_sources = len(ra_hyper) - len(hyper_indices_matched)
    false_sources_perc = false_sources/len(ra_hyper)*100.




    ### Plot part ###
    # Plot Flux Peak comparison
    flux_rms_to_plot = flux_peak_rms * 1.e6    # mJy -> muJy
    flux_hyper_to_plot = flux_peak_hyper * 1.e6    # mJy -> muJy


    plt.figure(figsize=(8, 6))
    plt.scatter(flux_rms_to_plot, flux_hyper_to_plot, color='blue', label=f'Catalog {i}')
    plt.plot([min(np.min(flux_rms_to_plot), np.min(flux_hyper_to_plot)), max(np.max(flux_rms_to_plot), np.max(flux_hyper_to_plot))], 
             [min(np.min(flux_rms_to_plot), np.min(flux_hyper_to_plot)), max(np.max(flux_rms_to_plot), np.max(flux_hyper_to_plot))], 'k--', label='1:1 Line')
    plt.xlabel('Flux Peak (RMS Gaussian) ($\mu$Jy)', fontsize=14)
    plt.ylabel('Flux Peak (Hyper Catalog) ($\mu$Jy)', fontsize=14)
    plt.title(f'Catalog {i} Flux Peak Comparison', fontsize=16)
    plt.legend(prop=legend_properties)
    # Customize axis ticks to be bold
    plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=16, width=1)  # Minor ticks
    
    # Make the tick labels bold
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Make the borders bold
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Remove the grid
    # plt.grid(False)
    
    # Save the plot
    plt.savefig(f'{dir_ps_out}Flux_peak_comparison_catalog_{i}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


    # Plot Integrated Flux comparison
    flux_rms_to_plot = flux_integrated_rms*1.e6    # mJy -> muJy
    flux_hyper_to_plot = flux_integrated_hyper*1.e6    # mJy -> muJy

    plt.figure(figsize=(8, 6))
    plt.scatter(flux_rms_to_plot, flux_hyper_to_plot, color='red', label=f'Catalog {i}')
    plt.plot([min(np.min(flux_rms_to_plot), np.min(flux_hyper_to_plot)), max(np.max(flux_rms_to_plot), np.max(flux_hyper_to_plot))], 
             [min(np.min(flux_rms_to_plot), np.min(flux_hyper_to_plot)), max(np.max(flux_rms_to_plot), np.max(flux_hyper_to_plot))], 'k--', label='1:1 Line')
    plt.xlabel('Integrated Flux (RMS Gaussian) ($\mu$Jy)', fontsize=14)
    plt.ylabel('Integrated Flux (Hyper Catalog) ($\mu$Jy)', fontsize=14)
    plt.title(f'Catalog {i} Integrated Flux Comparison', fontsize=16)
    plt.legend(prop=legend_properties)
    # Customize axis ticks to be bold
    plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=16, width=1)  # Minor ticks
    
    # Make the tick labels bold
    ax = plt.gca()
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontweight('bold')

    # Make the borders bold
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Remove the grid
    # plt.grid(False)

    # Save the plot
    plt.savefig(f'{dir_ps_out}Integrated_flux_comparison_catalog_{i}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


    # Print out some statistics
    print(f'Catalogue {i}:')
    print(f'Number of common sources: {len(flux_peak_rms)}')
    print(f'Number of false sources idenfied: {false_sources}, equivalent to {false_sources_perc:.1f} % of the total sources identified: {len(ra_hyper)} ')
    print(f'Minimum flux peak (RMS): {np.min(flux_peak_rms)} mJy, Max: {np.max(flux_peak_rms)} mJy')
    print(f'Minimum flux peak (Hyper): {np.min(flux_peak_hyper)} mJy, Max: {np.max(flux_peak_hyper)} mJy')
    print(f'Minimum integrated flux (RMS): {np.min(flux_integrated_rms)} mJy, Max: {np.max(flux_integrated_rms)} mJy')
    print(f'Minimum integrated flux (Hyper): {np.min(flux_integrated_hyper)} mJy, Max: {np.max(flux_integrated_hyper)} mJy')
    print('------------------------')



    ### save values in IPAC Table ###
    # Append values for each catalogue
    data.add_row([i, len(ra_rms), len(flux_peak_rms), len(ra_hyper), false_sources, round(false_sources_perc, 1)])
    
 
    
    # ################## Completeness Analysis ##################

    # Extract all RMS fluxes (for total counts) and matched RMS fluxes
    flux_int_rms_all = table_rms['Flux_Integrated']
    flux_int_rms_matched = table_rms['Flux_Integrated'][rms_indices_matched] #[min_indices[matches]]

    # flux_int_rms_all = table_rms['Flux_Peak']
    # flux_int_rms_matched = table_rms['Flux_Peak'][min_indices[matches]]

    # Define flux bins (adjust the range and step as needed)
    flux_bins = np.linspace(flux_int_rms_all.min(), flux_int_rms_all.max(), 15)


    # Initialize arrays to store results for each catalog
    bin_centers = []
    completeness = []


    # Compute completeness for each flux bin
    for flux_threshold in flux_bins:
        # Step 1: Filter RMS sources above the flux threshold
        rms_above_threshold_indices = np.where(flux_int_rms_all >= flux_threshold)[0]  # Indices of RMS sources above threshold
        total_rms_count = len(rms_above_threshold_indices)

        # Step 2: Filter the matches to only those corresponding to RMS sources above the threshold
        valid_rms_matches = np.intersect1d(rms_indices_matched, rms_above_threshold_indices)  # Correct intersection of matched RMS and threshold indices

        matched_hyper_count = len(valid_rms_matches)


       # Step 3: Calculate completeness
        completeness_ratio = matched_hyper_count / total_rms_count if total_rms_count > 0 else 0

        # Store results
        bin_centers.append(flux_threshold)
        completeness.append(completeness_ratio)


    # Store the results for plotting later
    all_completeness.append(completeness)
    all_bin_centers.append(bin_centers)


# Save the IPAC table
data.write(table_source_count, format='ascii.ipac', overwrite=True)
   




# ################## Plot All Completeness Curves ##################
plt.figure(figsize=(8, 6))

# List to store the flux values at 90% completeness for all curves
flux_at_90 = []

# Plot all the completeness curves with bolder lines
for i in range(tot_maps):
    scaled_flux_bins = np.array(all_bin_centers[i]) * 1.e6   # Jy -> µJy
    scaled_completeness = np.array(all_completeness[i]) * 100  # Scale completeness to percentage
    
    # Interpolate to find the flux at 90% completeness
    flux_90 = np.interp(90, scaled_completeness, scaled_flux_bins)  # Interpolate flux at 90%
    flux_at_90.append(flux_90)  # Append to the list
    
    # Plot the curve
    # plt.plot(scaled_flux_bins, scaled_completeness, label=f'Catalog {i+1}', linewidth=2.5)  # Increase line width
    plt.plot(scaled_flux_bins, scaled_completeness, linewidth=2.5)  # Increase line width

# Calculate the average flux at 90% completeness
average_flux_90 = np.mean(flux_at_90)

# Add horizontal line at 90% completeness
plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, label='90% Completeness')

# Add vertical line at the average 90% completeness flux
plt.axvline(x=average_flux_90, color='blue', linestyle='--', linewidth=1.5, label=f'Avg Int. Flux: {average_flux_90:.2f} µJy')

plt.xlabel('F$_{int}$ ($\mu$Jy)', fontsize=16)
plt.ylabel('Completeness (%)', fontsize=16)
# plt.title('Completeness as a Function of Peak Flux', fontsize=16, fontweight='bold')
plt.legend(prop=legend_properties)

# Customize axis ticks to be bold
plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=16, width=2)  # Minor ticks

# Make the tick labels bold
ax = plt.gca()
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontweight('bold')

# Make the borders bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Remove the grid
# plt.grid(False)

# Save the plot
plt.savefig(f'{dir_ps_out}Completeness_vs_Flux_Threshold.pdf', dpi=600, bbox_inches='tight')
plt.close()

#show plot
plt.show()





