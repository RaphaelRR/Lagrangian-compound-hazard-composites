"""parameters for compositing analyses"""
# Track selection parameters
valid_hours = [0,6,12,18] # Hours for which all data involved is available
year_range = [1980,2019]  # Range of years for which all data involved is available [1980,2019] for winds, precip, etc. Start in 2004 for dust

# Plotting and data storing parameters
half_width = 20 # Halfwidth of the interpolation region
res_factor = 8 # Factor for the resolution of interpolated data
coarse_factor = 4 # Factor for coarsening the data, either to smooth, or to block-average (Default is 4 for dynamical features)
eff_factor = int(res_factor/coarse_factor)

# Statistical analysis parameters
day_max_offset = 15 # Halfwidth of the time period centered on the event whithin which to seek random MC samples
n_samples = 2000 # Number of Monte-Carlo Samples to produce
alpha_fdr = 0.05 # Uncorrected threshold for statistical significance

# "plot" for plotting, or "sig" for significance testing
res_flag = 'sig'