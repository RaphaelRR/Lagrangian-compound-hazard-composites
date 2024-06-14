# -*- coding: utf-8 -*-
"""
Created on Thursday Sept 15 2022

@author: Raphael
"""

import numpy as np
import datetime
import pandas as pd
import xarray as xr
from scipy.signal import convolve2d
from scipy.io import loadmat
import multiprocessing
import params as pr
import os.path


def open_tracks():
#     # Datasets have [ID - Lon - Lat - year - month - day - hour - central pressure]
#     # Read COST action tracking dataset
#     fns = '~/COST_tracks/TRACKS1979_2020.TXT'
#     df_tracks = pd.read_csv(fns, sep='\s+',header=None)
#     # Read Jonathan and Shira classification data
#     fns = '~/COST_tracks/Events Clusters.csv'
#     df_clust = pd.read_csv(fns)

    # Tracks dataset
    tracks_mat = loadmat('/home/raphaelr/COST_tracks/Filtered_Tracks.mat')
    tmp = tracks_mat['Filtered_Tracks']
    # Resetting track numbering to be consistent with clusters
    for ii, track_ID in enumerate(np.unique(tmp[:,0])):
        tmp[tmp[:,0] == track_ID,0] = ii+1
    df_tracks = pd.DataFrame(tmp)
    df_tracks[0] = df_tracks[0].astype(int)
    df_tracks[3] = df_tracks[3].astype(int)
    df_tracks[4] = df_tracks[4].astype(int)
    df_tracks[5] = df_tracks[5].astype(int)
    df_tracks[6] = df_tracks[6].astype(int)
    # Clustering table
    ind_mat = loadmat('/home/raphaelr/COST_tracks/ind.mat')
    ind_mat = np.transpose(ind_mat['ind'])
    df_clust = pd.DataFrame(np.argwhere(ind_mat)+1)
    df_clust.columns = ['Event Number', 'Cluster Number']
    
    return df_tracks, df_clust



def get_storms(df_tracks,df_clust,cluster_number,n_storm):    
    # Get all events that fall within the selected cluster. If cluster_number==0: select all events
    if  isinstance(cluster_number,list):
        event_ID = np.array(df_clust.loc[df_clust['Cluster Number'].isin(cluster_number)]['Event Number'])
        df_select = df_tracks.loc[df_tracks[0].isin(event_ID)]
    else:
        if cluster_number != 0:
            event_ID = np.array(df_clust.loc[df_clust['Cluster Number']==cluster_number]['Event Number'])
            df_select = df_tracks.loc[df_tracks[0].isin(event_ID)] # Select all times of all events within a given cluster
        else: 
            event_ID = np.array(df_clust['Event Number'])
            df_select = df_tracks    
    # Extracting all lines where time of the day is compatible with whatever data we want to process
    df_select = df_select.loc[df_select[6].isin(pr.valid_hours)]    
    # Extract only the time of minimum pressure for all events
    # Get indices of min pressure
    pmin_ind = np.zeros(len(event_ID),'int64')
    aa = 0
    for ii in event_ID:
        pmin_ind[aa] = df_select.loc[df_select[0]==ii][7].idxmin()
        aa += 1       
    # Extract minimum pressure rows
    df_select = df_select.loc[pmin_ind]    
    # Second set of track selection criterion
    # Select only years that are included in the dust dataset
    df_select = df_select.loc[(df_select[3] >= pr.year_range[0])&(df_select[3] <= pr.year_range[1])]    
    # Reset index to make track selection easier. The event_ID is retained in df_test[0]
    # Note that tracks IDs increasing incrementally by 1
    df_select = df_select.reset_index(drop=True)    
    # If number of storms is specified to a non-zero value, else take all storms
    if n_storm != 0:
        df_select = df_select.loc[0:n_storm-1]    
    return df_select



def make_var_time(df_select):
    # List of row indices
    row_ind = df_select.index.values
    var_time = np.array([],dtype='datetime64')
    for tt in row_ind:
        yy = df_select.loc[tt,3]
        mm = df_select.loc[tt,4]
        dd = df_select.loc[tt,5]
        hh = df_select.loc[tt,6]
        # datetime format
        var_time = np.append(var_time, datetime.datetime(year=yy, month=mm, day=dd, hour=hh))
    return var_time



""" generate_fn_var creates a string that is the path to the file of the variable requested at the time requested """
""" var_name is the dataset name for the variable, var_time is a datetime object of the time requested """
def generate_fn_var(var_name,var_time):  
    if  (var_name=='duaod550')or(var_name=='pm10'):
        fn_var = '~/COST_tracks/CAMS_dust_movmean.nc'        
#     elif (var_name=='duaod550_mean')or(var_name=='pm10_mean'):
#         fn_var = '~/COST_tracks/CAMS_dust_mean_all.nc'  
    elif (var_name=='duaod550_mean')or(var_name=='pm10_mean'):
        fn_var = '~/COST_tracks/CAMS_dust_mean_seas.nc'       
    elif (var_name=='u10')or(var_name=='v10')or(var_name=='msl'):
        prefix_var = '/mnt/climstor/ecmwf/era5/raw/SFC/data/an_sfc_ERA5_'
        file_date = var_time.strftime('%Y-%m-%d')
        fn_var = prefix_var+file_date+'.nc'
    elif (var_name=='wind_mag'):
        prefix_var = '/scratch2/raphaelr/data/wind_mag/wmag'
        file_date = var_time.strftime('%Y')
        fn_var = prefix_var+file_date+'.nc'      
    elif (var_name=='precip')or(var_name=='precip_mask'): 
        prefix_var = '/scratch2/raphaelr/data/precip/precip'
        file_date = var_time.strftime('%Y')
        fn_var = prefix_var+file_date+'.nc'
    elif (var_name=='t2m'): 
        prefix_var = '/scratch2/raphaelr/data/t2m/t2m'
        file_date = var_time.strftime('%Y')
        fn_var = prefix_var+file_date+'.nc'
    elif var_name=='PV':
        prefix_var = '/mnt/climstor/ecmwf/era5/processed/PV/data/'
        file_date = var_time.strftime('%Y%m%d_%H')
        yy = var_time.strftime('%Y')
        fn_var = prefix_var+str(yy)+'/PV_'+file_date # NetCDF file without termination        
    elif var_name=='DI':
        prefix_var = '/scratch2/raphaelr/DI_data/work/meteogroup/DI/era5/grid_gt700/'
        file_date = var_time.strftime('%Y%m%d_%H')
        yy = var_time.strftime('%Y')
        fn_var = prefix_var+str(yy)+'/'+'di_gt700_'+file_date+'.cdf'        
    elif var_name=='WCB':
        prefix_var = '/scratch2/raphaelr/data/WCB/'
        file_date = var_time.strftime('%Y%m')
        folder_date = var_time.strftime('%Y%m%d_%H')
        fn_var = prefix_var+file_date+'/'+'T'+folder_date        
    elif (var_name=='cold_front')or(var_name=='warm_front'):
        fn_var = '~/front_data/era5_fronts_expanded_'+var_name[0:4]+'_1979--2020_mediterranean.nc'        
    elif var_name=='swh':
        fn_var = '~/COST_tracks/ERA5_wave_height.nc'    
    elif var_name=='fwi':
        prefix_var =  '/scratch2/raphaelr/data/forest_fire/era5/'
        yy = var_time.strftime('%Y')
        suffix_var = '_era5_fire_reorder.nc'
        fn_var = prefix_var+str(yy)+suffix_var    
    elif var_name=='fwi_mean':
        prefix_var =  '/scratch2/raphaelr/data/forest_fire/era5/'
        suffix_var = 'mean_fire_variables.nc'
        fn_var = prefix_var+suffix_var    
    else: print('Variable name is not known')        
    return fn_var



""" open_var returns the xarray dataset of the selected variable, within which the selected time is available"""
""" var_name and var_time are defined as in generate_fn_var, and fn_prev is the name of the previous dataset file opened"""
""" if fn_prev is empty or different from the output of generate_fn_var, reopen a dataset, if it is identical, do not reopen"""
def open_var(var_name,var_time,fn_prev,ds_prev):   
    fn_var = generate_fn_var(var_name,var_time)
    if fn_var != fn_prev:
        ds_var = xr.open_dataset(fn_var)
    else:    
        ds_var = ds_prev    
    # return the xarray dataset and the path to the associated file
    return ds_var, fn_var



""" read_var returns the interpolated matrix to composite """
def read_var(ds_var,var_name,var_time,lon_qvec,lat_qvec,ds_mask):      
    if (var_name=='duaod550')or(var_name=='pm10'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        var_interp = ds_var[var_name].sel(time=field_date).interp(longitude=lon_qvec,latitude=lat_qvec).data        
#     elif (var_name=='duaod550_mean')or(var_name=='pm10_mean'):
#         var_interp = ds_var[var_name[0:-5]].interp(longitude=lon_qvec,latitude=lat_qvec).data  
    elif (var_name=='duaod550_mean')or(var_name=='pm10_mean'):
        mm = int(var_time.strftime('%-m'))
        var_interp = ds_var[var_name[0:-5]].sel(month=mm).interp(longitude=lon_qvec,latitude=lat_qvec).data       
    elif (var_name=='u10')or(var_name=='v10')or(var_name=='msl'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        var_interp = ds_var[var_name].sel(time=field_date).interp(lon=lon_qvec,lat=lat_qvec).data        
    elif (var_name=='precip'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        var_interp = ds_var['TP'].sel(time=field_date).interp(lon=lon_qvec,lat=lat_qvec).data
    elif (var_name=='precip_mask'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        tmp_mask = ds_mask.interp(longitude=lon_qvec,latitude=lat_qvec).data
        var_interp = ds_var['TP'].sel(time=field_date).interp(lon=lon_qvec,lat=lat_qvec).data*tmp_mask
        # Set entire array to Nan if storm is not over  water (not for fire case)
        if np.isnan(ds_mask.interp(longitude=lon_qvec[int(len(lon_qvec)/2)],latitude=lat_qvec[int(len(lat_qvec)/2)]).data): 
            var_interp[:,:] = np.nan  
    elif (var_name=='t2m'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        var_interp = ds_var[var_name].sel(time=field_date).interp(lon=lon_qvec,lat=lat_qvec).data    
    elif (var_name=='wind_mag'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        tmp_var_name = '__xarray_dataarray_variable__'
        var_interp = ds_var[tmp_var_name].sel(time=field_date).interp(lon=lon_qvec,lat=lat_qvec).data       
    elif var_name=='PV':
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S')
        var_interp = ds_var[var_name].sel(time=field_date,lev=[5.0,6.0,7.0,8.0,9.0]).interp(lon=lon_qvec,lat=lat_qvec).mean(dim='lev').data 
    elif var_name=='DI':
        lat_DI = np.linspace(ds_var.domymin,ds_var.domymax,ds_var['N'].shape[2])
        lon_DI = np.linspace(ds_var.domxmin,ds_var.domxmax,ds_var['N'].shape[3])           
        # Change coordinates so they can be indexed with latitude (dimy_N) and longitude (dimx_N) vector
        ds_var = ds_var.assign_coords(dimy_N=("dimy_N", lat_DI),dimx_N=("dimx_N", lon_DI))
        var_interp = np.squeeze(ds_var['N'].interp(dimx_N=lon_qvec,dimy_N=lat_qvec).data)
    elif var_name=='WCB':
        tmp1 = ds_var['GT800'].interp(lon=lon_qvec,lat=lat_qvec).data
        tmp2 = ds_var['MIDTROP'].interp(lon=lon_qvec,lat=lat_qvec).data 
        var_interp = np.squeeze(tmp1+tmp2)  
    elif (var_name=='cold_front')or(var_name=='warm_front'):
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000')
        var_interp = ds_var['fronts'].sel(time=field_date).interp(longitude=lon_qvec,latitude=lat_qvec,method='nearest').data
        var_interp[~np.isnan(var_interp)]=1
        var_interp[np.isnan(var_interp)]=0  
    elif var_name=='swh':
        field_date = var_time.strftime('%Y-%m-%dT%H:%M:%S'+'.000000000') 
        tmp_mask = ds_mask.interp(longitude=lon_qvec,latitude=lat_qvec).data
        var_interp = ds_var[var_name].sel(time=field_date).interp(longitude=lon_qvec,latitude=lat_qvec).data*tmp_mask
        # Set entire array to Nan if storm is not over  water (not for fire case)
        if np.isnan(ds_mask.interp(longitude=lon_qvec[int(len(lon_qvec)/2)],latitude=lat_qvec[int(len(lat_qvec)/2)]).data): 
            var_interp[:,:] = np.nan    
    elif var_name=='fwi':
        field_date = var_time.strftime('%Y-%m-%dT'+'00:00:00.000000000') 
        tmp_mask = ds_mask.interp(longitude=lon_qvec,latitude=lat_qvec).data
        var_interp = ds_var[var_name].sel(time=field_date).interp(longitude=lon_qvec,latitude=lat_qvec).data*tmp_mask     
    elif var_name=='fwi_mean':
        mm = int(var_time.strftime('%-m'))
        tmp_mask = ds_mask.interp(longitude=lon_qvec,latitude=lat_qvec).data
        var_interp = ds_var['fwi'].sel(month=mm).interp(longitude=lon_qvec,latitude=lat_qvec).data*tmp_mask
    else: print('Variable name is not known')
    return var_interp



""" read_thresh returns the interpolated matrix to composite, subjected to a threshold, returns a boolean matrix """
def apply_thresh(var_name,var_interp,lon_qvec,lat_qvec):       
    # For hazards, the initial threshold will be fixed
    if (var_name=='wind_mag'):
        # Thresholding
        ds_thresh = xr.open_dataset('~/process_data/percentiles_wmag.nc')
        var_thresh = ds_thresh['__xarray_dataarray_variable__'].sel(quantile=0.98).interp(lon=lon_qvec,lat=lat_qvec,method='nearest').data
        var_thresh = np.maximum(var_thresh,10)
        var_interp = (var_interp>=var_thresh)
    elif (var_name=='precip')or(var_name=='precip_mask'):
        # Thresholding
        ds_thresh = xr.open_dataset('~/process_data/percentiles_precip.nc')
        var_thresh = ds_thresh['TP'].sel(quantile=0.99).interp(lon=lon_qvec,lat=lat_qvec,method='nearest').data
        var_thresh = np.maximum(var_thresh,0.001)
        var_interp = (var_interp>=var_thresh)
    elif (var_name=='t2m'):
        # Thresholding
        var_thresh = 30 # [C]
        var_interp = (var_interp-273.15>=var_thresh)
    elif var_name=='swh':
        # Thresholding 
        var_thresh = 4 # [m]
        var_interp = (var_interp>=var_thresh)        
    elif var_name=='duaod550':
        # Thresholding 
        var_thresh = 0.5 # [m]
        var_interp = (var_interp>=var_thresh)    
    # For dynamical features, the initial threshold will be just an existence
    # Perhaps some smoothing may be added at this stage.
    elif var_name=='pm10':
    # Thresholding 
        var_thresh = 50*1e-9 # [m]
        var_interp = (var_interp>=var_thresh)    
    # For dynamical features, the initial threshold will be just an existence
    # Perhaps some smoothing may be added at this stage.
    elif var_name=='DI':
        # Smoothing and Booleaning
        var_interp = smooth(var_interp, pr.eff_factor)
        var_interp = np.array(var_interp,dtype=bool)
    elif var_name=='WCB':
        # Smoothing and Booleaning
        var_interp = np.array(var_interp,dtype=bool)        
    elif (var_name=='cold_front')or(var_name=='warm_front'):
        # Smoothing and Booleaning
#         var_interp = smooth(var_interp, pr.eff_factor)
        var_interp = np.array(var_interp,dtype=bool)            
    else: print('Variable name is not known or has no defined threshold')   
    return var_interp



def make_mask(var_name,var_time):
    if (var_name=='swh')or(var_name=='fwi')or(var_name=='precip_mask'):
  
        # Vectors for creating the mask
        if var_name=='precip_mask':
            ds_var, fn_var = open_var('swh',var_time[0],[],[]) 
            lon = ds_var['longitude'].data
            lat = ds_var['latitude'].data
            tmp_time = ds_var['time'].data
            # Make the mask        
            ds_mask = ds_var['swh'].sel(time=tmp_time[0]) 
        else:
            # Open variables for mask
            ds_var, fn_var = open_var(var_name,var_time[0],[],[]) 
            lon = ds_var['longitude'].data
            lat = ds_var['latitude'].data
            tmp_time = ds_var['time'].data
            # Make the mask        
            ds_mask = ds_var[var_name].sel(time=tmp_time[0]) 
            
        if (var_name=='swh')or(var_name=='precip_mask'):
            # Mask out Atlantic Ocean and red sea
            ds_mask[:,lon<-6]=float("nan")
            ds_mask[lat<29,:]=float("nan")
            ds_mask[lat>48,:]=float("nan")
            ds_mask[lat>43,lon<-1]=float("nan")
            ds_mask = ds_mask/ds_mask
        elif var_name=='fwi':        
            # Mask out Sahara and North Africa, which will high-bias the fire risk even though there is little combustible
            ds_mask[lat<31,:]=float("nan")
            ds_mask[lat<34,lon<34]=float("nan")
            ds_mask[lat<=35.75,lon<0]=float("nan")
            ds_mask[lat<=37.5,(lon>=0)*(lon<11.75)]=float("nan")           
            ds_mask = ds_mask/ds_mask
    else: ds_mask = []
    return ds_mask

def make_thresh(var_name):          
    if var_name=='precip':
        ds_thresh = xr.open_dataset('~/process_data/percentiles_precip.nc')
    else: ds_thresh = []
    return ds_thresh



def composite_var(var_name,df_select,var_time,ds_mask,anom_flag):           
    # Initialize variables
    ds_var = []
    fn_var = []
    ds_var_mean = []
    fn_var_mean = []
    if pr.res_flag=='sig':
        var_comp = np.zeros((pr.eff_factor*pr.half_width+1,pr.eff_factor*pr.half_width+1,len(var_time)),dtype=bool)
    elif pr.res_flag=='plot':
        var_comp = np.zeros((pr.res_factor*pr.half_width+1,pr.res_factor*pr.half_width+1,len(var_time)),dtype=bool)
    # Iterate through list of points
    for aa in np.arange(0,len(var_time)):
        # Query vectors
        lon_cent = df_select.loc[aa,1]
        lat_cent = df_select.loc[aa,2]
        lon_qvec = np.linspace(lon_cent-pr.half_width,lon_cent+pr.half_width,pr.half_width*pr.res_factor+1)
        lat_qvec = np.linspace(lat_cent-pr.half_width,lat_cent+pr.half_width,pr.half_width*pr.res_factor+1)
        # Open and read variables      
        ds_var, fn_var = open_var(var_name,var_time[aa],fn_var,ds_var)
        var_interp = read_var(ds_var,var_name,var_time[aa],lon_qvec,lat_qvec,ds_mask) 
        # Compute density (coarsened)
        tmp_dens = ~np.isnan(var_interp)
        if pr.res_flag=='sig':
            tmp_dens =  (block_average(tmp_dens,pr.coarse_factor) > 0)
        if aa == 0: 
            dens_map = 1*tmp_dens
        else:
            dens_map = dens_map + 1*tmp_dens           
        # Compute anomaly if indicated   
        if anom_flag == 1:
            var_name_mean = var_name + '_mean'
            ds_var_mean, fn_var_mean = open_var(var_name_mean,var_time[aa],fn_var_mean,ds_var_mean)
            var_interp_mean = read_var(ds_var_mean,var_name_mean,var_time[aa],lon_qvec,lat_qvec,ds_mask)             
            var_interp += -var_interp_mean
        # Apply Thresholding and smoothing to produce boolean arrays
        var_interp = apply_thresh(var_name,var_interp,lon_qvec,lat_qvec) 
        if pr.res_flag=='sig':
            # Block average and restate logical condition
            var_interp = block_average(var_interp,pr.coarse_factor)
            var_comp[:,:,aa] = (var_interp>0)
        elif pr.res_flag=='plot':
            var_comp[:,:,aa] = var_interp
    # Averaging                   
    return var_comp, dens_map



# Function to generate Monte-Carlo sample
# Input df_select is a pandas dataframe
# Output mc_datetime is an array of datetime objects
def mc_make_var_time(df_select,n_seed):    
    row_ind = df_select.index.values 
    n_storm = len(df_select.index.values)    
    # Specified random seed, so that the n_th monte-carlo composite for any variable, will account for at all the same points
    np.random.seed(n_seed)    
    # Random years for test sample
    rand_year = np.random.randint(pr.year_range[0]+1, high=pr.year_range[1]-1, size=n_storm, dtype=int)
    # Monte-Carlo datetime
    mc_var_time = np.array([],dtype='datetime64')
    # First, replace year by a random year
    for tt in row_ind:
        yy = rand_year[tt]
        mm = df_select.loc[tt,4]
        dd = df_select.loc[tt,5]
        hh = df_select.loc[tt,6]
        # ERA5 time format
        if (mm==2)&(dd==29): dd=28
        mc_var_time = np.append(mc_var_time, datetime.datetime(year=yy, month=mm, day=dd, hour=hh))
    # Then offset by a random number of days, and hours (remaining where data is defined)
    off_day = np.random.randint(-pr.day_max_offset+1, high=pr.day_max_offset-1, size=n_storm, dtype=int) + 1/4*np.random.randint(-3, high=4, size=n_storm, dtype=int)
    off_day = off_day*datetime.timedelta(days=1)
    mc_var_time += off_day    
    return mc_var_time

#----------------------printing version--------------------------------#

""" n_samples is the number of monte carlo composites to produce"""
def mc_composite_var(var_name,df_select,ds_mask,anom_flag,n_proc,prefix_dir,cluster_number):             
    data=() # Initialize run tuple
    for aa in np.arange(0,pr.n_samples):
        data += ([aa,var_name,df_select,ds_mask,anom_flag,prefix_dir,cluster_number],) # Append to tuple 
    a_pool = multiprocessing.Pool(n_proc) # Init. multiprocessing 
    a_pool.map(mc_single_composite_var, data) # Run multiprocessing  
    return


def mc_single_composite_var(data):    
    # Initialize data
    aa = data[0]
    var_name =  data[1]
    df_select =  data[2]
    ds_mask =  data[3]
    anom_flag =  data[4]
    prefix_dir =  data[5]
    cluster_number = data[6]
    # Make composite
    mc_var_time = mc_make_var_time(df_select,aa)        
    var_comp, placeholder = composite_var(var_name,df_select,mc_var_time,ds_mask,anom_flag)
    fn = prefix_dir+var_name+'/'+'MC_'+var_name+'_bool_c'+str(cluster_number)+'_'+str(aa)+'.npy'
    np.save(fn, var_comp)
    return

def mix_and_match(*bool_arrays,dens):
    # Initialize array
    mm_bool = bool_arrays[0]
    # Intersect with other arrays, if provided
    for ii in np.arange(1,len(bool_arrays)):
        # Select the shortest length to compare arrays
        len1 = mm_bool.shape[2]
        len2 = bool_arrays[ii].shape[2]
        eff_len = np.minimum(len1,len2)
        mm_bool = np.logical_and(mm_bool[:,:,-eff_len:], bool_arrays[ii][:,:,-eff_len:])
    mm_comp = np.sum(mm_bool,axis=2)/dens
    return mm_comp
#----------------------end printing version--------------------------------#

# """ n_samples is the number of monte carlo composites to produce"""
# def mc_composite_var(var_name,df_select,ds_mask,anom_flag,n_proc):             
#     data=() # Initialize run tuple
#     for aa in np.arange(0,pr.n_samples):
#         data += ([aa,var_name,df_select,ds_mask,anom_flag],) # Append to tuple 
#     a_pool = multiprocessing.Pool(n_proc) # Init. multiprocessing 
#     tmp = a_pool.map(mc_single_composite_var, data) # Run multiprocessing  
#     mc_var_comp = np.concatenate(tmp,axis=3) # Prepare output    
#     return mc_var_comp



# def mc_single_composite_var(data):    
#     # Initialize data
#     aa = data[0]
#     var_name =  data[1]
#     df_select =  data[2]
#     ds_mask =  data[3]
#     anom_flag =  data[4]
#     # Make composite
#     mc_var_time = mc_make_var_time(df_select,aa)        
#     var_comp, placeholder = composite_var(var_name,df_select,mc_var_time,ds_mask,anom_flag)
#     out_comp = np.expand_dims(var_comp, np.ndim(var_comp)) # Add a dimension for concatenation to MC composites
#     return out_comp


# def mix_and_match(bool_array_1,bool_array_2,point_density):
#     bool_intersection = np.logical_and(bool_array_1, bool_array_2)
#     if np.ndim(bool_array_1)==4: point_density = np.expand_dims(point_density,2)
#     mm_comp = np.sum(bool_intersection,axis=2)/point_density
#     return mm_comp


def fdr_test(comp,mc_comp):    
    n_comp = mc_comp.shape[2]    
    # Compute threshold for significance
    # Compute empirical p-value as one minus the sum of MC local means smaller than the storm composite local mean values, 
    # divided by the number of MC composites
    pvals = 1 - np.sum(np.greater(np.expand_dims(comp, 2),mc_comp),2)/n_comp
    # Sort p_values
    pvals_sort = np.sort(np.reshape(pvals,(pvals.shape[0]*pvals.shape[1])))
    # Produce the i vector and the correspoding FDR threshold vector
    fdr_thresh = pr.alpha_fdr*np.arange(1,len(pvals_sort)+1)/len(pvals_sort)
    # Vector of points that are significant
    sig_vec = pvals_sort<=fdr_thresh
    # Compute effective FDR p_value
    if any(sig_vec): p_fdr = np.max(pvals_sort[sig_vec])
    else: p_fdr = fdr_thresh[0]     
    # Evaluate where data is statistically significant
    sig_map = pvals<p_fdr    
    return sig_map, p_fdr


def smooth(y, box_pts):
    box = np.ones((box_pts,box_pts))/box_pts**2
    y_smooth = convolve2d(y, box, mode='same',boundary='symm')
    return y_smooth



def block_average(var,factor):
    var_coarse = np.add.reduceat(np.add.reduceat(var, np.arange(0, var.shape[0], factor), axis=0),np.arange(0, var.shape[1], factor), axis=1)/factor**2
    return var_coarse


""" Post processing functions"""

# The true mix and match function
# Takes as input two lists: 
# One list of variable names (e.g., var_list = ['precip','wind_mag'])
# One list of mix-and-match operations (e.g., mm_list = [1,[1,2],2])
# In addition, it takes the density matrix of the samples, the directory to the stored data, and the cluster number
def make_mc_composite(var_list,mm_list,dens,prefix_dir,cluster_number):    
    # Initialize output variable
    output_list = []
    for mm in mm_list:
        output_list.append(np.zeros((dens[0].shape[0],dens[0].shape[1],pr.n_samples)))        
    # Iterate through all MC samples
    for mc_ID in np.arange(0,pr.n_samples):      
        # Load one set of corresponding samples for all variables
        input_list = []
        for var_name in var_list:
            fn = prefix_dir+var_name+'/'+'MC_'+var_name+'_bool_c'+str(cluster_number)+'_'+str(mc_ID)+'.npy'
            input_list.append(np.squeeze(np.load(fn)))        
        # Mix and match to populate the output list
        for ii, mm in enumerate(mm_list):
            # Make a list to input to mix_and_match(), which will be unpacked with operator *
            mm_input = list(map(input_list.__getitem__, mm))                        
            output_list[ii][:,:,mc_ID] = mix_and_match(*mm_input,dens=dens[ii])
    return output_list  


# The single compositing function
def make_composite(var_list,mm_list,prefix_dir,cluster_number,flag):    
    # Initialize output variable
    output_list_dens = []
    input_list_dens = []
    output_list = []
    input_list = []
    for var_name in var_list:
        if flag=='plot':
            fn = prefix_dir+var_name+'/'+var_name+'_hr_bool_c'+str(cluster_number)+'.npz'
        elif flag=='sig':
            fn = prefix_dir+var_name+'/'+var_name+'_bool_c'+str(cluster_number)+'.npz'        
        tmp = np.load(fn)
        input_list.append(tmp['comp'])
        input_list_dens.append(tmp['dens'])      
    for ii, mm in enumerate(mm_list):
            # Make a list to input to mix_and_match(), which will be unpacked with operator *
            mm_input = list(map(input_list.__getitem__, mm))
            mm_input_dens = list(map(input_list_dens.__getitem__, mm))
            output_list_dens.append(np.min(np.array(mm_input_dens),axis=0))
            output_list.append(mix_and_match(*mm_input,dens=output_list_dens[ii])) # IMPORTANT! MAKE SURE TO CHOOSE DENS WHEN SWH IS USED!        
    return output_list, output_list_dens 


# Evaluate statistical significance
def compute_sig(mm_comp_list,mm_mc_comp_list):
    output_list = []
    for ii, mm_comp in enumerate(mm_comp_list):
        mm_comp_sig, p_fdr = fdr_test(mm_comp,mm_mc_comp_list[ii])
        output_list.append(mm_comp_sig)
    return output_list