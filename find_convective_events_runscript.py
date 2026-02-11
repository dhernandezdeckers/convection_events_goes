import numpy as np
import pickle
import pdb
from joblib import Parallel, delayed
from convective_events_func import *
from extras import *
import gc
import os

"""
This script is the second step for convective event identification based on IR 
data as presented in:
    Hernandez-Deckers, D.. (2022) Features of atmospheric deep convection
    in northwestern South America obtained from infrared satellite data.
    Q J R Meteorol Soc, 148( 742), 338– 350. https://doi.org/10.1002/qj.4208

This script should be run after running the script read_GOES_data_runscript.py,
so that the files T_grid**.npy, time_**.npy and area_**.npy must be present in
the folder CONVECTION_'case_name'.

After running this code, use the output written in CONVECTION_'case_name' folder 
to investigate the temporal and spatial distribution of convective events:
    N_events_total_*.npy': distribution of storms taking into account the entire area of BT<T_min at all time steps.
    N_events_total_Tmin_*.npy: distribution of storms taking into account the entire area of BT<T_min only at the peak time of the events.
    N_events_hh_*.npy: number of events by hour, counting events at all gridboxes with BT<Tmin during peak of event only.
    N_events_mm_*.npy: number of events by month, counting events at all gridboxes with BT<Tmin during peak of event only.
    mean_ssize_Tmin_*.npy: mean storm size, where individual storm sizes have been assigned to the area below BT<Tmin at event peak only (excludes sizes > max_sizekm2).
    mean_sdur_Tmin_*.npy: mean storm duration, where individual storm duration has been assigned to the area below BT<Tmin at event peak only (excludes sizes > max_sizekm2).
    N_events_nxny*.npy: total number of events over each gridbox regardless of TRMM precipitation data. Each event is counted only at one gridbox (where it reaches minimum BT).
    N_events_wTRMM_nxny*.npy: total number of events over each gridbox. Each event is counted only at one gridbox (where it reaches minimum BT).
    N_events_wTRMM_sizelimit_nxny*.npy: total number of events over each gridbox excluding sizes larger than max_sizekm2. Each event is counted only at one gridbox (where it reaches minimum BT).
    N_events_wTRMM_mindTpos_nxny*.npy: same as N_events_wTRMM_nxny*.npy, but each event is counted at the gridbox where the steepest BT decrease was observed.
    events_nxny*.txt: text file with list of events. Rows are: 
    DATE_MIN_T        LON LAT MIN_T       DATE_MAX_DT       LON LAT MAX_DT(K/h)
The script will also try to create two sample plots. If it succeeds, these are:
    event_rate_density_size_duration.png: convective event size and duration distributions.
    ssize_duration_distr.png: spatial distribution of event rate density, size and duration (requires cartopy module).

This code is free to use with no warranty. For reference, please cite:

    Hernandez-Deckers, D.. (2022) Features of atmospheric deep convection
    in northwestern South America obtained from infrared satellite data.
    Q J R Meteorol Soc, 148( 742), 338– 350. https://doi.org/10.1002/qj.4208

D. Hernández-Deckers - 2022
dhernandezd@unal.edu.co
"""

#**********************************************************
# Parameters for convective event identification (from namelist.txt):
#**********************************************************
param_dict = read_namelist_parameters()
case_name       = param_dict['case_name']  
T_min           = float(param_dict['T_min'])
T_minmin        = float(param_dict['T_minmin'])
dTmin2h         = float(param_dict['dTmin2h'])
dt_max          = float(param_dict['dt_max'])
max_sizekm2     = float(param_dict['max_sizekm2'])
min_TRMM_precip = float(param_dict['min_TRMM_precip'])
TRMM_data_path  = param_dict['TRMM_data_path']
njobs           = int(param_dict['njobs'])
UTC_offset      = -int(param_dict['UTC'])
Ea_r            = float(param_dict['Ea_r'])

# ************************************************************************************
# ***************** END OF PARAMTER READING ******************************************
# ************************************************************************************


folder='CONVECTION_'+case_name
if not os.path.exists(folder):
    print('folder '+folder+' does not exist!')
    print('You must first run the sript read_GOES_data_runscript.py!!')
    print("If yes, check 'case_name' and run again.")
else:
    fname = os.popen('ls '+folder+'/T_grid_'+case_name+'.npy').read().split()
    if len(fname)==1:
        T_grid = np.load(fname[0])
        fn = os.popen('ls '+folder+'/time_'+case_name+'.npy').read().strip()
        time = np.load(fn)
        fn = os.popen('ls '+folder+'/area_'+case_name+'.p').read().strip()
        area = pickle.load( open(fn, 'rb'), encoding='latin1')
        nx = area.nx
        ny = area.ny
    elif len(fname)>1:
        print('Several grids are present:')
        for i in range(len(fname)):
            print(fname)
        print('Please enter values for nx and ny:')
        nx = int(input('nx= '))
        ny = int(input('ny= '))
        T_grid  = np.load(folder+'/T_grid_'+case_name+'.npy')
        time    = np.load(folder+'/time_'+case_name+'.npy')
        area    = pickle.load( open(folder+'/area_'+case_name+'.p','rb'),encoding='latin1')
    print(case_name)
    print('\ngridboxes are %.2f x %.2f km'%(area.dx,area.dy))
    
    #np.warnings.filterwarnings('ignore')
    
    T_grid[np.where(np.isnan(T_grid))]=9999
    
    # ************************************************************************************
    # select all instants in which at least one gridbox reaches a temperature below T_minmin
    T_minmin_ind=[]
    for i in range(time.shape[0]):
        if np.any(T_grid[i,:,:]<T_minmin):
            T_minmin_ind.append(i)
    T_minmin_ind=np.asarray(T_minmin_ind)
    
    # ************************************************************************************
    # In order to parallelize, group "consecutive" (max 1 hour gaps) images:
    sub_T_minmin_ind = [] 
    counter=0
    while counter<len(T_minmin_ind):
        temp=[]
        temp.append(T_minmin_ind[counter])
        #while counter<len(T_minmin_ind)-1 and T_minmin_ind[counter+1]<=T_minmin_ind[counter]+dt_max/(deltat/60.):
        while counter<len(T_minmin_ind)-1 and (dt.datetime(*time[T_minmin_ind[counter+1]])-dt.datetime(*time[T_minmin_ind[counter]])<=dt.timedelta(hours=dt_max)):
            temp.append(T_minmin_ind[counter+1])
            counter+=1
        sub_T_minmin_ind.append(temp)
        counter+=1
    
    # ************************************************************************************
    # jobs are splitted where there is a gap of more than 1 hour without possible convective events in the region
    jobs=[]
    for i in range(len(sub_T_minmin_ind)):
        jobs.append((T_grid[sub_T_minmin_ind[i]],time[sub_T_minmin_ind[i]],area,i,len(sub_T_minmin_ind),T_minmin,T_min,min_TRMM_precip,TRMM_data_path,dt_max,UTC_offset))
    
    if min_TRMM_precip>0:
        print('Will read 3-hourly TRMM precipitation data')
    # ************************************************************************************
    # Possible events are identified as those with a contiguous region of BT below T_min 
    # with at least one point with BT below T_minmin (each event is an object):
    print('Running %d jobs (please be patient)...'%(len(jobs)))
    out=Parallel(n_jobs=njobs,pre_dispatch=njobs)(delayed(find_events)(*jobs[i]) for i in range(len(jobs)))
    # Beware: this can take long!
    print('Done!! Now will find steepest BT decrease of each possible event...')
    del jobs
    
    gc.collect()

    events=[]
    for i in range(len(out)):
        events.extend(out[i])
    

    i_ev = 0
    for event in events:
        """
        For each event, find the largest decrease in brightness temperature (BT) during 2 hours
        within a window of up to 3 hours before the corresponding peak:
        """
        print('%04d/%d'%(i_ev+1,len(events)),end='\r')
        dT=[]
        dT_t=[]
        dT_coord=[]
        #assign a unique identifier (int):
        event.id = i_ev
        #for each event, record the steepest T decrease in 2h in any of the peaks up to 3 hours before the corresponding peak:
        for i in range(len(event.peaks)):
            ind_lon=np.where(np.round(area.lon_centers,3)==event.peaks[i][0])[0][0]
            ind_lat=np.where(np.round(area.lat_centers,3)==event.peaks[i][1])[1][0]
            ttmp=event.t[i]
            ind_t=np.where((time[:,0]==ttmp[0])*(time[:,1]==ttmp[1])*(time[:,2]==ttmp[2])*(time[:,3]==ttmp[3])*(time[:,4]==ttmp[4])*(time[:,5]==ttmp[5]))[0][0]
            ind_t_time = dt.datetime(*time[ind_t])
            ind_tmin3 = ind_t
            while dt.datetime(*time[ind_tmin3])>(dt.datetime(*time[ind_t])-dt.timedelta(hours=3)) and ind_tmin3>0:
                ind_tmin3+=-1
            #T_gridbox_series=T_grid[np.max([0,int(ind_t-3./(deltat/60.))]):ind_t+1,ind_lon,ind_lat]
            T_gridbox_series=T_grid[ind_tmin3:ind_t+1,ind_lon,ind_lat]
            time_gridbox_series = time[ind_tmin3:ind_t+1]
            dT2h=0.
            dT2h_t=ind_t
            dT2h_coord=(ind_lon,ind_lat)
            jf=len(T_gridbox_series)-1
            j0=jf
            while (dt.datetime(*time_gridbox_series[j0])>dt.datetime(*time_gridbox_series[jf])-dt.timedelta(hours=2)) and j0>0:
                j0+=-1
            #j0=jf-int(2./(deltat/60.))
            #count=0
            while j0>=0:
                if 9999 in [T_gridbox_series[jf],T_gridbox_series[j0]]:
                    dT2h_temp=0
                else:
                    dT2h_temp=T_gridbox_series[jf]-T_gridbox_series[j0]
                if dT2h_temp<dT2h:
                    dT2h = dT2h_temp
                    j_tmp = jf
                    counter=0
                    while (dt.datetime(*time_gridbox_series[j_tmp])>dt.datetime(*time_gridbox_series[jf])-dt.timedelta(hours=1)) and j_tmp>0:
                        j_tmp+=-1
                        counter+=1
                    dT2h_t = ind_t - counter
                    #dT2h_t=ind_t-count-int(1./(deltat/60.))
                    dT2h_coord=(ind_lon,ind_lat)
                #count+=1
                jf=jf-1
                j0=jf
                while (dt.datetime(*time_gridbox_series[j0])>dt.datetime(*time_gridbox_series[jf])-dt.timedelta(hours=2)) and j0>0:
                    j0+=-1
                #j0=j0-1
                #jf=jf-1
            dT.append(dT2h)
            dT_t.append(dT2h_t)
            dT_coord.append(dT2h_coord)
        ind_dT=np.where(np.asarray(dT)==np.min(np.asarray(dT)))[0][0]
        event.add_dT2h(dT[ind_dT],time[dT_t[ind_dT]],dT_coord[ind_dT])
        i_ev+=1
    print('\nDone!')

    # ************************************************************************************
    # compute velocity of events:
    print('\nComputing velocity of all convective events...')
    for event in events:
        event.find_event_vel(Ea_r=Ea_r)
    print('Done!')
    
    # ************************************************************************************
    # save the essential data of all events as a numpy array and write it to a file:
    data, coords_data = compile_events(events,time,area.lon_centers,area.lat_centers)
    np.save(folder+'/data_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),data)
        
    # select events with rapid decrease in BT (<dTmin2h) and minimum TRMM precip (if min_TRMM_precip is set to >0)
    ind_events=np.where((data[:,8]<=dTmin2h)*(data[:,16]>=min_TRMM_precip))[0] 
    events_valid = np.asarray(events)[ind_events]
    
    print('\nwriting csv file...')
    #write_events_file_from_data(data, area, ind_events, fname=folder+'/events_'+case_name+'_Tmin%d_T2min%d.txt'(,T_minmin,T_min))
    write_events_file_from_data_csv(data, area, ind_events, fname=folder+'/events_'+case_name+'_Tmin%d_T2min%d.csv'%(T_min,T_minmin))
    print('Done!')

    # plot size and duration distributions:
    ssize = data[ind_events][:,17]*1e-5
    sdur  = data[ind_events][:,18]
    plot_ssize_duration_distr( ssize, sdur, folder )
    
    print('\nComputing spatial and temporal distribution of events...')
    # ************************************************************************************
    # get the coordinates of the events in a particular format that is designed to determine
    # spatial distributions:
    events_coords, events_coords_data = get_events_coordinates( events_valid, ind_events )
    del events
    gc.collect()

    # ************************************************************************************
    # count the number of events in each gridbox (in total, by hour, by month, etc.)
    N_events_total      = np.zeros([nx,ny])     # this will include all time steps of the events
    N_events_hh         = np.zeros([24,nx,ny])  # this includes only the peak time step of the events (by hour)
    N_events_mm         = np.zeros([12,nx,ny])  # this includes only the peak time step of the events (by month)
    N_events_total_Tmin = np.zeros([nx,ny])     # like N_events_total, but only during the peak time step (probably more useful)
    mean_ssize_Tmin     = np.zeros([nx,ny])     # mean size of the event (assigned to all gridboxes with BT<T_min at peak of event)
    mean_sdur_Tmin      = np.zeros([nx,ny])     # mean duration of the event assigned to all gridboxes with BT<T_min at peak of event. 
    mean_ssize_MM_Tmin  = np.zeros([12,nx,ny])     # mean size of the event (assigned to all gridboxes with BT<T_min at peak of event) by month
    
    jobs=[]
    if np.mod(len(events_coords_data),njobs)==0:
        nevents_job = int(len(events_coords_data)/njobs)
    else:
        nevents_job = int(len(events_coords_data)/njobs)+1
    i_events=0
    ijobs=0
    while((ijobs+1)*nevents_job<=len(events_coords_data)):
        i0=ijobs*nevents_job
        i1=np.min([len(events_coords_data),(ijobs+1)*nevents_job])
        jobs.append(( events_coords[events_coords_data[i0][2]:events_coords_data[i1-1][-1]+1], events_coords_data[i0:i1], events_coords_data[ijobs*nevents_job][2], area.lon_corners, area.lat_corners, area.mask, nx, ny ))
        ijobs+=1
    out = Parallel(n_jobs=len(jobs))(delayed(count_events)(*jobs[i]) for i in range(len(jobs)))

    for i in range(len(jobs)):
        N_events_total      = N_events_total + out[i][0]
        N_events_hh         = N_events_hh + out[i][1]
        N_events_mm         = N_events_mm + out[i][2]
        N_events_total_Tmin = N_events_total_Tmin + out[i][3]
        mean_ssize_Tmin     = mean_ssize_Tmin + out[i][4]
        mean_sdur_Tmin      = mean_sdur_Tmin + out[i][5]
        mean_ssize_MM_Tmin  = mean_ssize_MM_Tmin + out[i][6]
    mean_ssize_Tmin = (mean_ssize_Tmin/N_events_total_Tmin)*area.dx*area.dy # to get the average size in km2
    mean_ssize_MM_Tmin = (mean_ssize_MM_Tmin/N_events_mm)*area.dx*area.dy
    #mean_sdur_Tmin  = mean_sdur_Tmin*(deltat/60.)/N_events_total_Tmin       # to get the average duration in hours
    mean_sdur_Tmin  = (mean_sdur_Tmin/60)/N_events_total_Tmin       # to get the average duration in hours
    del jobs
    gc.collect()

    # Now count each event only at one gridpoint: where it reaches its minimum BT.
    # compute this by splitting the grid through x:
    out=Parallel(n_jobs=np.min([nx,njobs]))(delayed(run_events_ny)(*[data,ix,ny,area.lon_corners,area.lat_corners,area.mask,dTmin2h,min_TRMM_precip,max_sizekm2]) for ix in range(nx))
    N_events                    = np.zeros([nx,ny]) # convective events regardless of TRMM precip (counting only point of peak intensity) 
    N_events_wTRMM              = np.zeros([nx,ny]) # convective events considering TRMM precip (counting only point of peak intensity)
    N_events_wTRMM_sizelimit    = np.zeros([nx,ny]) # convective events considering TRMM precip and a maximum size (counting only point of peak intensity)
    N_events_wTRMM_mindTpos     = np.zeros([nx,ny]) # convective events considering TRMM, but located at steepest BT decrease location (counting only point of peak intensity)
    # Now merge the nx jobs:
    for ix in range(nx):
        N_events[ix,:]                  = out[ix][0]
        N_events_wTRMM[ix,:]            = out[ix][1]
        N_events_wTRMM_sizelimit[ix,:]  = out[ix][2]
        N_events_wTRMM_mindTpos[ix,:]   = out[ix][3]
    
    # Apply the mask (if any) to all the arrays:
    if np.any(area.mask==0):
        outside=np.where(area.mask==0)
        N_events[outside]                   = np.nan
        N_events_wTRMM[outside]             = np.nan
        N_events_wTRMM_sizelimit[outside]   = np.nan
        N_events_wTRMM_mindTpos[outside]    = np.nan
        N_events_total[outside]             = np.nan
        N_events_total_Tmin[outside]        = np.nan
        for i in range(N_events_hh.shape[0]):
            N_events_hh[i][outside]         = np.nan   
        for i in range(N_events_mm.shape[0]):
            N_events_mm[i][outside]         = np.nan   
    
    # Compute mean and median of storm duration, but assign this only to the gridpoint of minimum BT of each event:
    mean_sdur_minBTpeak     = np.zeros([nx,ny])
    median_sdur_minBTpeak   = np.zeros([nx,ny])
    mean_ssize_minBTpeak    = np.zeros([nx,ny])
    median_ssize_minBTpeak  = np.zeros([nx,ny])
    
    njobs_tmp=np.min([24,njobs])
    njobs_iy=int(ny/np.min([24,njobs]))+1
    out=Parallel(n_jobs=njobs_tmp)(delayed(get_sdursize)(*(np.arange(nx),np.arange(i*njobs_iy,np.min([(i+1)*njobs_iy,ny])),nx,ny,data,area.lon_centers,area.lat_centers,dTmin2h,min_TRMM_precip,max_sizekm2)) for i in range(njobs_tmp))
    for i in range(njobs_tmp):
        mean_sdur_minBTpeak +=out[i][0]
        median_sdur_minBTpeak +=out[i][1]
        mean_ssize_minBTpeak +=out[i][2]
        median_ssize_minBTpeak +=out[i][3]

    # Compute mean and median of storm duration by month, but assign this only to the gridpoint of minimum BT of each event:
    mean_sdur_minBTpeak_mm     = np.zeros([12,nx,ny])
    median_sdur_minBTpeak_mm   = np.zeros([12,nx,ny])
    mean_ssize_minBTpeak_mm    = np.zeros([12,nx,ny])
    median_ssize_minBTpeak_mm  = np.zeros([12,nx,ny])
    
    njobs_tmp=np.min([24,njobs])
    njobs_iy=int(ny/np.min([24,njobs]))+1
    out=Parallel(n_jobs=njobs_tmp)(delayed(get_sdursize_mm)(*(np.arange(nx),np.arange(i*njobs_iy,np.min([(i+1)*njobs_iy,ny])),nx,ny,data,area.lon_centers,area.lat_centers,dTmin2h,min_TRMM_precip,max_sizekm2)) for i in range(njobs_tmp))
    for i in range(njobs_tmp):
        mean_sdur_minBTpeak_mm +=out[i][0]
        median_sdur_minBTpeak_mm +=out[i][1]
        mean_ssize_minBTpeak_mm +=out[i][2]
        median_ssize_minBTpeak_mm +=out[i][3]
    del data
    gc.collect()

    # This distribution of storms takes into account the "cumulative" size throughout the entire storm lifetime (includes all area of BT<T_min at all time steps) (probably not very useful):
    np.save(folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_total)
    
    # This contains the number of events by hour, counting events at all gridboxes with BT<Tmin during peak of event only:
    np.save(folder+'/N_events_hh_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_hh)
    
    # This contains the number of events by month, counting events at all gridboxes with BT<Tmin during peak of event only:
    np.save(folder+'/N_events_mm_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_mm)
    
    # This contains the total number of events counting events at all gridboxes with BT<Tmin during peak of event only (probably more useful than N_events_total):
    np.save(folder+'/N_events_total_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_total_Tmin)
    
    # This contains the mean storm size, where individual storm sizes have been assigned to the area below BT<Tmin at event peak only
    np.save(folder+'/mean_ssize_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_ssize_Tmin)
    
    # This contains the mean storm size by month, where individual storm sizes have been assigned to the area below BT<Tmin at event peak only
    np.save(folder+'/mean_ssize_MM_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_ssize_MM_Tmin)

    # This contains the mean storm duration, where individual storm duration has been assigned to the area below BT<Tmin at event peak only
    np.save(folder+'/mean_sdur_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_sdur_Tmin)
    
    # The following files count events only at one gridbox per event (where it reaches its minimum BT):
    np.save(folder+'/N_events_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events)                 # all events
    np.save(folder+'/N_events_wTRMM_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_wTRMM)          # considering TRMM precip
    np.save(folder+'/N_events_wTRMM_sizelimit_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_wTRMM_sizelimit)# considering TRMM precip and maximum size
    np.save(folder+'/N_events_wTRMM_mindTpos_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),N_events_wTRMM_mindTpos) # considering TRMM precip, but located at the steepest BT decrease location instead of at minimum BT location.
    
    np.save(folder+'/mean_ssize_minBTpeak_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_ssize_minBTpeak) 
    np.save(folder+'/mean_sdur_minBTpeak_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_sdur_minBTpeak) 

    np.save(folder+'/mean_ssize_minBTpeak_mm_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_ssize_minBTpeak_mm) 
    np.save(folder+'/mean_sdur_minBTpeak_mm_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin),mean_sdur_minBTpeak_mm) 
    
    # Make a plot similar to Fig. 1 in Hernandez-Deckers (2022):
    try:
        make_sample_plot(area,N_events_total_Tmin,N_events_wTRMM,mean_ssize_minBTpeak,mean_sdur_minBTpeak,nx,ny,folder, T_grid, time)
        print('created sample spatial distribution plot!')    
    except:
        print('could not create sample spatial distribution plot!')
    del T_grid
    gc.collect()

    print('\nCreating NETCDF file... (this is the last step and may take very long! all other plots and outputs are ready)')
    #*******************************************************************************
    # save netcdf file with all convection events information:
    save_events_data_netcdf4(area, events_valid, time, fname=folder+'/conv_events_'+case_name+'_Tmin%d_T2min%d.nc'%(T_min,T_minmin),njobs=njobs)
    print('Done!')
    del time
    del area
    del events_valid
    gc.collect()
    print('\n***************\nFinished!\n')

