import numpy as np
import datetime as dt
import pdb
import events_obj
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from pylab import *

def find_events(T_grid, time, area, njob, njobs, T_minmin, T_min, min_TRMM_precip, TRMM_data_path, dt_max, UTC_offset):
    masks=[]
    counter=0
    events=[]
    tf_events=np.array([])
    dx=area.dx
    dy=area.dy

    for i in range(T_grid.shape[0]): # in each image that has at least one gridbox with T<T_minmin, identify all contiguous regions with T<T_min
        systems=np.ones_like(T_grid[i,:,:])*np.nan #this matrix will have identifiers (numbers) at every gridbox that is part of a convective system, nans elsewhere.

        ind=np.where(T_grid[i,:,:]<T_min)  # regions with low brightness temperatures, but not necessarily lower than T_minmin (these are broader regions that will indicate the entire system).
        for ii in range(len(ind[0])):
            systems[ind[0][ii],ind[1][ii]]=ii+1 # assign a number to each gridbox with T<T_min
        for ii in range(len(ind[0])):
            for jj in range(len(ind[0])):
                if contiguous((ind[0][ii],ind[1][ii]),(ind[0][jj],ind[1][jj])):
                    if systems[ind[0][ii],ind[1][ii]]<systems[ind[0][jj],ind[1][jj]]:   # if two gridboxes are contiguous, the same number is assigned to them, to identify the same system.
                        systems[np.where(systems==systems[ind[0][jj],ind[1][jj]])]=systems[ind[0][ii],ind[1][ii]]
                    else:
                        systems[np.where(systems==systems[ind[0][ii],ind[1][ii]])]=systems[ind[0][jj],ind[1][jj]]
        for ii in range(len(ind[0])):
            system=np.where(systems==ii+1)
            if not np.any(T_grid[i,system[0],system[1]]<T_minmin):
                systems[system]=np.nan # remove any system that does not have at least one gridbox with T<T_minmin.
        nmin=np.nanmin(systems)

        systems=systems-nmin+1  # make sure the first number assigned is 1
        ii=2
        system=np.where(systems==ii)
        while np.nanmax(systems)>=ii:   # make all others numbers assigned to be ascending without gaps
            system=np.where(systems==ii)
            while len(system[0])==0 and np.nanmax(systems)>ii:
                systems2=np.copy(systems)
                systems2[np.where(np.isnan(systems2))]=-999
                systems[np.where(systems2>=ii)]+=-1
                system=np.where(systems==ii)
            ii+=1
        T_peak=np.ones_like(systems)*np.nan
        nmax=np.nanmax(systems)
        for ii in range(int(nmax)):
            system=np.where(systems==ii+1) # coordinates of points with low brightness temperatures
            peak_ind=np.where(T_grid[i,:,:]==np.nanmin(T_grid[i,:,:][system])) # identify the location of the minimum T of a system
            if len(peak_ind[0])>1: # this is unlikely. If it happens, the first point is used.
                if len(peak_ind[0])==2:
                    if ((peak_ind[0][1]-peak_ind[0][0])**2+(peak_ind[1][1]-peak_ind[1][0])**2)<=2:
                        print('Warning! (%d) adjacent gridboxes have the exact same minimum low brightness temperature (%.1f K)... The first point will be used. (not a serious concern)'%(len(peak_ind[0]),np.nanmin(T_grid[i,:,:][system])))
                    else:
                        print('Warning! (%d) gridboxes %.1f km apart have the exact same minimum low brightness temperature (%.1f K)... The first point will be used. (a bit weird)'%(len(peak_ind[0]),np.sqrt((peak_ind[0][1]-peak_ind[0][0])**2+(peak_ind[1][1]-peak_ind[1][0])**2)*dx,np.nanmin(T_grid[i,:,:][system])))
                else:
                    print('Warning! (%d) gridboxes have the exact same minimum low brightness temperature... The first point will be used. (strange!)'%(len(peak_ind[0])))
            T_peak[peak_ind]=T_grid[i,:,:][peak_ind] # (nx,ny) grid with the peak (minimum) temperature value of the system at its position, the rest nans.
            coords=[]
            for point in range(len(system[0])):
                coords.append((np.round(area.lon_centers[system][point],3),np.round(area.lat_centers[system][point],3))) # saves all coordinates of system
            # creates a "temporary" event with this first time step 
            if min_TRMM_precip>0:
                event0=events_obj.Event(t0=time[i], tf=time[i], coords=coords, peaks=(np.round(area.lon_centers[peak_ind][0],3),np.round(area.lat_centers[peak_ind][0],3),T_peak[peak_ind][0]),dx=dx,dy=dy,TRMM=True, TRMM_data_path=TRMM_data_path,UTC_offset=UTC_offset)
            else:
                event0=events_obj.Event(t0=time[i], tf=time[i], coords=coords, peaks=(np.round(area.lon_centers[peak_ind][0],3),np.round(area.lat_centers[peak_ind][0],3),T_peak[peak_ind][0]),dx=dx,dy=dy,TRMM=False)
            if counter==0: # if this is the first image, this event can't be part of an earlier event, so it is automatically included as an "official" new event
                events.append(event0)
                tf_events=np.append(tf_events,event0.tf_hrs)
            else: # if this is not the first time step, it could be part of an earlier event, so it should be merged with that one:
                merge=False
                events_to_merge = np.where(tf_events>=(event0.tf_hrs-dt_max))[0]    # choose all events that finish up to dt_max hours before the current event. These are possible candidates for being an earlier stage of this same event!
                ord_ind=np.flip(np.argsort(tf_events[events_to_merge]))             # check first the closest (in time) events to the current event
                nevent=0
                while not merge and nevent<len(events_to_merge):
                    merge=compare_events(new_event=event0,old_event=events[events_to_merge[nevent]],dt_max=dt_max)    # check if the two events have a minimum overlap (see compare_events for details), and similar sizes
                    if merge:
                        events[events_to_merge[nevent]].add_system(event0)                              # if yes, add the current time step to the previous event
                        tf_events[events_to_merge[nevent]]=events[events_to_merge[nevent]].tf_hrs
                    nevent+=1
                if not merge:   # if not, create a new event
                    events.append(event0)
                    tf_events=np.append(tf_events,event0.tf_hrs)
        masks.append(systems)
        
        counter+=1
    print('finished job # %04d (of %d)'%(njob+1,njobs), end='\r')
    return events

def compare_events(new_event, old_event, dt_max=1): 
    """
    To determine if two events that are separated in time actually correspond to the same event,
    they must match in at least a fraction of gridboxes, which depends on how far in time the events 
    are: the threshold is computed as the minimum between 1/(10*t_diff) and 0.5, where t_diff is the 
    time separation in hours; so, for a 1h separation, at least 10% of the gridboxes of the smaller 
    event should match. For a 15min separation, the threshold is 40%. For a 5 minute interval, we 
    use 0.5 (50% overlap). Also, sizes should be similar (up to 50% difference). This last condition 
    assumes that a system will not change its size dramatically even after the maximum of dt_max (1 hour)
    """
    matches=0
    #make sure the two events are not separated by more than dt_max hours:
    t_diff = (dt.datetime(*new_event.t0)-dt.datetime(*old_event.tf)).total_seconds()/3600. # time gap between last time of old event and first time of new event, in hours
    if 0<=t_diff<=dt_max:
        if len(new_event.coords[0])<len(old_event.coords[-1]):
            for i in range(len(new_event.coords[0])):
                if new_event.coords[0][i] in old_event.coords[-1]:
                    matches+=1
        else:
            for i in range(len(old_event.coords[-1])):
                if old_event.coords[-1][i] in new_event.coords[0]:
                    matches+=1
        total1 = np.min([len(old_event.coords[-1]),len(new_event.coords[0])]) #size of smallest system
        total2 = np.max([len(old_event.coords[-1]),len(new_event.coords[0])]) #size of largest system
        #if matches/total1>=0.1 and (np.abs(len(old_event.coords[-1])-len(new_event.coords[0]))<=0.5*total2): old version
        if (t_diff==0 and matches/total1>=0.5 and (np.abs(len(old_event.coords[-1])-len(new_event.coords[0]))<=0.5*total2)):
            return True
        elif t_diff>0 and (matches/total1>=np.min([(1/(10*t_diff)),0.5])) and (np.abs(len(old_event.coords[-1])-len(new_event.coords[0]))<=0.5*total2):
            return True
        else:
            return False
    else:
        return False

def compile_events(events,time,lon_centers,lat_centers):
    """
    Creates a numpy array called "data" that contains all the information of all events. This can be saved to a file.
    data will be an array where each row contains the essential information of each identified convective event
    """
    data=[]
    coords=[]
    for event in events:
        event.find_minT()
        ind=event.minT_ind
        t=event.dT_t
        t0 = event.t0
        tf = event.tf
        duration = (dt.datetime(*tf)-dt.datetime(*t0)).total_seconds()/3600.
        # storm size is estimated at minT_ind time (area of region below Tmin at time of minT)
        # Tmin lon lat yy mm dd hr min maxdT lon(maxdT) lat(maxdT) yy(maxdT) mm dd hr min max_TRMM_precip area(of system at Tmin time, in km2) duration(hours)
        data.append([event.peaks[ind][2], event.peaks[ind][0], event.peaks[ind][1], event.t[ind][0], event.t[ind][1], event.t[ind][2], event.t[ind][3], event.t[ind][4], event.dT, lon_centers[int(event.dT_coord[0])][0], lat_centers[0,int(event.dT_coord[1])], t[0], t[1], t[2], t[3], t[4], np.nanmax(event.TRMM_precip), len(event.coords[ind])*event.area, duration])
        coords.append(event.coords)
    return np.asarray(data), coords

def write_events_file_from_data(data, area, valid_indices, fname='event_list.txt'):
    # ************************************************************************************
    # Write list of events to a text file, including timing and location of minimum BT, and 
    # timing and location of its steepests decrease in BT. If all events in data are to be 
    # used, valid_indices should be an array with all indces (e.g., np.arange(data.shape[0]))
    #*************************************************************************************

    f=open(fname,'w')
    f.write('DATE_MIN_T        LON LAT MIN_T       DATE_MAX_DT       LON LAT MAX_DT(K/h)\n')
    
    for i in range(data.shape[0]):
        if i in valid_indices:
            t_minT      = data[i,3:8]
            t_maxdT     = data[i,11:16]
            minT        = data[i,0]
            lon_minT    = data[i,1]
            lat_minT    = data[i,2]
            maxdT       = data[i,8]
            lon_maxdT   = data[i,9]
            lat_maxdT   = data[i,10]
            if area.mask[np.where((np.around(area.lon_centers,decimals=3)==lon_minT)*(np.around(area.lat_centers,decimals=3)==lat_minT))][0]==1:
                f.write('%04d %02d %02d %02d %02d'%(t_minT[0],t_minT[1],t_minT[2],t_minT[3],t_minT[4])+' %.3f %.3f %.1f '%(lon_minT,lat_minT,minT)+' %04d %02d %02d %02d %02d'%(t_maxdT[0],t_maxdT[1],t_maxdT[2],t_maxdT[3],t_maxdT[4])+' %.3f %.3f %.1f\n'%(lon_maxdT,lat_maxdT,maxdT))
    f.close()
    print('Created file '+fname+' with list of events.')



def count_events( events_coords, events_coords_data, ind0, lon_corners, lat_corners, mask, nx, ny ):
    """
    Count events at each gridbox based on different criteria, but taking into account the size of the event
    as the area with BT<T_min.
    """
    # Each event is counted here at all gridboxes where BT<Tmin
    N_events_total = np.zeros([nx,ny])      # this will include all time steps of the events
    N_events_hh = np.zeros([24,nx,ny])      # this includes only the peak time step of the events (by hour)
    N_events_mm = np.zeros([12,nx,ny])      # this includes only the peak time step of the events (by month)
    N_events_total_Tmin = np.zeros([nx,ny]) # like N_events_total, but only during the peak time step (probably more useful)
    mean_ssize_Tmin = np.zeros([nx,ny])     # mean size of the event at peak of event (assigned to all gridboxes with BT<T_min at peak of event)
    mean_sdur_Tmin = np.zeros([nx,ny])      # mean duration of the event assigned to all gridboxes with BT<T_min at peak of event.
     
    for ind in range(len(events_coords_data)):
        n_coords = np.sum(events_coords_data[ind][4:-4]) # number of coordinates (gridpoints) in event (total)
        all_coords = events_coords[events_coords_data[ind][2]-ind0:events_coords_data[ind][2]-ind0+n_coords]
        n_coords_before_Tmin = int(np.sum(events_coords_data[ind][4:4+events_coords_data[ind][-4]]))   #number of coordinates before starting those at Tmin (of this event)
        coords_Tmin_startind = events_coords_data[ind][2]-ind0+n_coords_before_Tmin
        coords_Tmin_stopind = coords_Tmin_startind + events_coords_data[ind][4+events_coords_data[ind][-4]]
        coords_Tmin = events_coords[coords_Tmin_startind:coords_Tmin_stopind] # these are the coordinates of all gridboxes covered by event (BT<T_min) at peak of event only
        counted_coords=[]
        # ****************************************************
        # Here all time steps of the event are used:
        for coords in all_coords:
            if not ((coords[0],coords[1]) in counted_coords): # each location should be counted once per event, even if it appears in several timesteps!
                indlon=np.where(lon_corners<coords[0])[0][-1]
                indlat=np.where(lat_corners<coords[1])[1][-1]
                if mask[indlon,indlat]==1:
                    N_events_total[indlon,indlat]+=1          
                    counted_coords.append((coords[0],coords[1]))

        # ****************************************************
        # Here only the time step of the event peak is used (when BT peaks)
        counted_coords=[]
        for coords in coords_Tmin:                         
            if not ((coords[0],coords[1]) in counted_coords):
                indlon=np.where(lon_corners<coords[0])[0][-1]
                indlat=np.where(lat_corners<coords[1])[1][-1]
                if mask[indlon,indlat]==1:
                    counted_coords.append((coords[0],coords[1]))
                    N_events_total_Tmin[indlon,indlat]+=1
                    N_events_hh[events_coords_data[ind][-3],indlon,indlat]+=1
                    N_events_mm[events_coords_data[ind][-2]-1,indlon,indlat]+=1
                    mean_ssize_Tmin[indlon,indlat] += len(coords_Tmin) # still need to multiply times dx*dy (and divide by N_events_total_Tmin
                    mean_sdur_Tmin[indlon,indlat] += events_coords_data[ind][3] # THIS IS NOW DURATION IN MINUTES! (IGNORE THIS:still need to multiply times dt (e.g., 0.5h) and divide by N_events_total_Tmin)
    return N_events_total, N_events_hh, N_events_mm, N_events_total_Tmin, mean_ssize_Tmin, mean_sdur_Tmin

def contiguous(ind1, ind2):
    """
    return True if indices ind1 and ind2 are 
    contiguous, False otherwise.
    (ind1 and ind2 should be tuples)
    """
    i1=ind1[0]
    j1=ind1[1]
    i2=ind2[0]
    j2=ind2[1]
    if np.abs(i2-i1)<=1 and np.abs(j2-j1)<=1:
        return True
    else:
        return False

def run_events_ny(data, ix, ny, lon_corners, lat_corners, mask, dTmin2h, min_TRMM_precip, max_sizekm2):
    """
    As opposed to count_events, this function counts each event only at the (grid)point where it 
    reaches its minimum BT. It is designed for parallellization in the x direction.
    """
    N_events1=np.ones(ny)*np.nan
    N_events                    = np.ones(ny)*np.nan # convective events regardless of TRMM precip
    N_events_wTRMM              = np.ones(ny)*np.nan # convective events considering TRMM precip
    N_events_wTRMM_sizelimit    = np.ones(ny)*np.nan # convective events considering TRMM precip and a maximum size
    N_events_wTRMM_mindTpos     = np.ones(ny)*np.nan # convective events considering TRMM, but located at steepest BT decrease location
    for iy in range(ny):
        if mask[ix,iy]==1:
            ind_cases1=np.where((lon_corners[ix,0]<=data[:,1])*(data[:,1]<=lon_corners[ix+1,0])*(lat_corners[0,iy]<=data[:,2])*(data[:,2]<=lat_corners[0,iy+1])*(data[:,8]<=dTmin2h))[0]
            ind_cases2=np.where((lon_corners[ix,0]<=data[:,1])*(data[:,1]<=lon_corners[ix+1,0])*(lat_corners[0,iy]<=data[:,2])*(data[:,2]<=lat_corners[0,iy+1])*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
            ind_cases3=np.where((lon_corners[ix,0]<=data[:,1])*(data[:,1]<=lon_corners[ix+1,0])*(lat_corners[0,iy]<=data[:,2])*(data[:,2]<=lat_corners[0,iy+1])*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h)*(data[:,17]<max_sizekm2))[0]
            ind_cases4=np.where((lon_corners[ix,0]<=data[:,9])*(data[:,9]<=lon_corners[ix+1,0])*(lat_corners[0,iy]<=data[:,10])*(data[:,10]<=lat_corners[0,iy+1])*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0] #location of mindT instead of Tmin

            N_events[iy]                = len(ind_cases1)
            N_events_wTRMM[iy]          = len(ind_cases2)
            N_events_wTRMM_sizelimit[iy]= len(ind_cases3)
            N_events_wTRMM_mindTpos[iy] = len(ind_cases4)
    return N_events, N_events_wTRMM, N_events_wTRMM_sizelimit, N_events_wTRMM_mindTpos

def get_events_coordinates( events_valid, ind_events ):
    """
    From a list of events (events_valid), this function returns the coordinates 
    of all gridpoints of the events (events_coords) and a key file that allows 
    to read those coordinates knowing to which time step of the event they 
    correspond (events_coords_data)
    """
    events_coords = []
    events_coords_data = []
    ievent=0
    ind_coords_all = 0
    for event in events_valid:
        edata=[]
        edata.append(ievent)                                 # 0. index of event
        edata.append(ind_events[ievent])                     # 1. index of event in the data array
        edata.append(ind_coords_all)                         # 2. index in events_coords where coords of this event start
        edata.append(int((dt.datetime(*event.tf)-dt.datetime(*event.t0)).total_seconds()/60)) # 3. event duration in minutes
        #edata.append(len(event.coords))                      # 3. number of timesteps n (duration)
        for it in range(len(event.coords)):                
            edata.append(len(event.coords[it]))              # 4,5,.. 3+n. number of gridpoints in each timestep (as many elements as number of timesteps (2) 
            for i_gridbox in range(len(event.coords[it])):
                events_coords.append([event.coords[it][i_gridbox]])
            ind_coords_all+=len(event.coords[it])
        event.find_minT()
        edata.append(event.minT_ind)                         # 4+n.(-4) timestep of peak (min) BT
        edata.append(event.minT[3])                          # 5+n.(-3) local time of peak (min) BT
        edata.append(event.minT[1])                          # 6+n.(-2) month of event
        edata.append(ind_coords_all-1)                       # 7+n.(-1) last index in events_coords where coords of this event finish
        events_coords_data.append(edata)
        ievent+=1
    events_coords=np.asarray(events_coords).squeeze()
    events_coords_data=np.asarray(events_coords_data, dtype=object)
    return events_coords, events_coords_data 

def get_sdursize(ix,iy,nx,ny,data,lon_centers,lat_centers,dTmin2h,min_TRMM_precip,max_sizekm2):
    """
    Calculate event duration and size (with maximum size max_sizekm2) and assign it only to the 
    point of the minimum BT (minBT peak)
    """
    mean_sdur_minBTpeak     = np.zeros([nx,ny])
    median_sdur_minBTpeak   = np.zeros([nx,ny])
    mean_ssize_minBTpeak    = np.zeros([nx,ny])
    median_ssize_minBTpeak  = np.zeros([nx,ny])
    for i in ix:
        for j in iy:
            lon=lon_centers[i,j]
            lat=lat_centers[i,j]
            ind=np.where((data[:,1]==np.round(lon,3))*(data[:,2]==np.round(lat,3))*(data[:,8]<=dTmin2h)*(data[:,16]>=min_TRMM_precip)*(data[:,17]<max_sizekm2))
            if len(ind[0])>=3:
                mean_sdur_minBTpeak[i,j]    = np.nanmean(data[ind][:,18])
                mean_ssize_minBTpeak[i,j]   = np.nanmean(data[ind][:,17])
                median_sdur_minBTpeak[i,j]  = np.nanmedian(data[ind][:,18])
                median_ssize_minBTpeak[i,j] = np.nanmedian(data[ind][:,17])
    return mean_sdur_minBTpeak, median_sdur_minBTpeak, mean_ssize_minBTpeak, median_ssize_minBTpeak

def plot_ssize_duration_distr( ssize, sdur, folder='.'):
    """
    Make a plot of the distributions of size and duration of all events
    """
    plt.figure( figsize=(8,3) )
    gs = gridspec.GridSpec( 1, 2, left=0.1, right=0.99, hspace=0.2, wspace=0.05, top=0.96, bottom=0.2 )
    ax = subplot(gs[0])
    plt.hist(ssize,bins=100,range=(0,9),log=True)
    plt.ylabel('# of events',fontsize=12)
    plt.xlabel('event area (x10$^5$ km$^2$)',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim(0.5,1e5)
    plt.xlim(0,9)

    ax = subplot(gs[1])
    plt.hist(sdur,bins=100,range=(0,90),log=True)
    plt.xlabel('event duration (h)',fontsize=12)
    plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    plt.ylim(0.5,1e5)
    ax.yaxis.set_major_formatter( NullFormatter() )
    plt.xlim(0,90)
    plt.savefig(folder+'/ssize_duration_distr.png')

def get_factor_available_data(T_grid,time,nx,ny,threshold=0.8,delta_t=30):
    """
    computes the number of 'useful' images relative to the maximum possible by hour.
    This is used to correct distributions in case there is a systematic difference
    in available images at certain hours. Useful means that at least 'threshold' fraction
    of gridcells are non nan.
    """
    t0=dt.datetime(*time[0])
    tf=dt.datetime(*time[-1])
    max_nimgs_hh=(tf-t0).total_seconds()*(60/delta_t)/(24*3600) # maximum number of images corresponding to one hour in the entire period assuming 30 minute intervals
    valid_fraction=[]
    interruptions = np.zeros([24])
    for i in range(T_grid.shape[0]):
        valid_fraction.append(1.-(len(np.where(np.isnan(T_grid[i]))[0]))/(nx*ny))
    valid_fraction=np.asarray(valid_fraction)
    n_imgs = np.ones([24])*np.nan
    for hh in range(24):
        n_imgs[hh]=len(np.where((time[:,3]==hh)*(valid_fraction>=threshold))[0])
    time_factor_hh = n_imgs/max_nimgs_hh
    tot_time_factor = np.sum(n_imgs)/(24*max_nimgs_hh) # maximum number of images in the entire period
    return time_factor_hh, tot_time_factor

def make_sample_plot(area,N_events_total_Tmin,N_events_wTRMM,mean_ssize_minBTpeak,mean_sdur_minBTpeak,nx,ny,folder, T_grid, time):
    """
    Create plot of event rate density (a) considering the entire region with BT<T_min, 
    (b) considering only the location of minimum BT, (c) mean event size and (d) mean event duration.
    """
    from scipy.ndimage import gaussian_filter
    import cartopy.crs as ccrs
    t_length_years = (dt.datetime(*time[-1])-dt.datetime(*time[0])).total_seconds()/(365*24*3600)

    time_factor_hh, tot_time_factor = get_factor_available_data(T_grid, time, nx, ny)
    fig=plt.figure(figsize=(14,5.9))
    gs = gridspec.GridSpec(1, 4, left=0.035, right=0.99, hspace=0.2, wspace=0.05, top=0.99, bottom=0.13)
    ax = subplot(gs[0],projection=ccrs.Mercator(central_longitude=-75))
    cs=plot_image_cartopy(area, ax, N_events_total_Tmin[:,:]*10/(tot_time_factor*t_length_years*area.dx*area.dy),vmin=0,vmax=10, ticks=np.arange(0,10,2), cmap='afmhot_r',label='event rate density (BT<235K)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='a)',lllat=area.lrlat, urlat=area.urlat,lllon=area.ullon,urlon=area.urlon) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
    ax = subplot(gs[1],projection=ccrs.Mercator(central_longitude=-75))
    # smooth with a gaussian filter helps visualization:
    N_events_wTRMM_smooth = gaussian_filter(N_events_wTRMM, sigma=1, mode='reflect', cval=0.0 )
    N_events_wTRMM_smooth[-2,:]=np.nan
    N_events_wTRMM_smooth[:,1]=np.nan
    cs=plot_image_cartopy(area, ax, N_events_wTRMM_smooth[:,:]*10/(tot_time_factor*t_length_years*area.dx*area.dy),vmin=0,vmax=0.275, ticks=np.arange(0,0.4,0.1), cmap='afmhot_r',label='event rate density (Tmin location)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='b)',labelslat=False,lllat=area.lrlat, urlat=area.urlat,lllon=area.ullon,urlon=area.urlon) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
    ax = subplot(gs[2],projection=ccrs.Mercator(central_longitude=-75))
    # smooth with a gaussian filter helps visualization:
    ssize_smooth = gaussian_filter(mean_ssize_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
    plot_image_cartopy(area, ax, ssize_smooth*1e-3, vmin=0, vmax=38, cmap='afmhot_r', label='mean event size (x10$^3$km$^2$))', remove_borders=True, ticks=[0,10,20,30],title='c)',logscale=False, labelslat=False,lllat=area.lrlat, urlat=area.urlat,lllon=area.ullon,urlon=area.urlon)
    ax = subplot(gs[3],projection=ccrs.Mercator(central_longitude=-75))
    sdur_smooth = gaussian_filter(mean_sdur_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
    plot_image_cartopy(area, ax, sdur_smooth, vmin=0, vmax=6, cmap='afmhot_r', label='mean event duration (h)', remove_borders=True, ticks=[0,1,2,3,4,5,6],title='d)',logscale=False, labelslat=False, lllat=area.lrlat, urlat=area.urlat,lllon=area.ullon,urlon=area.urlon)
    plt.savefig(folder+'/event_rate_density_size_duration.png',dpi=300)

def plot_image_cartopy( area, ax, T, time=[2011,1,1,0,0], cmap='Greys', vmin=200,vmax=300,lllat=-2.15,urlat=12.7,lllon=-79.95,urlon=-68.9,cb=True,topo=False, remove_borders=False,label='(K)',title=None,ticks=[200,300],corners=[np.nan, np.nan, np.nan, np.nan],logscale=False, labelslon=True, labelslat=True, loncorners=None,latcorners=None,extend='max',boxes=None,boxescolor=['k'],fslonlat=11,dl=3):
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    lat_0 = dl*int(lllat/dl)
    lat_f = dl*int(urlat/dl)
    dl = 3
    lon_0 = dl*int(lllon/dl)
    lon_f = dl*int(urlon/dl)
    extent = [lllon,urlon,lllat,urlat]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m',linewidth=0.5,color='k')
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.3,color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(lon_0,lon_f+1,dl))
    gl.ylocator = mticker.FixedLocator(np.arange(lat_0,lat_f+1,dl))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': fslonlat}
    gl.xlabel_style = {'size': fslonlat}
    projection = ccrs.LambertConformal(central_latitude=5, central_longitude=-75)

    if (not labelslon):
        gl.bottom_labels = False
    if (not labelslat):
        gl.left_labels=False
    if topo:
        try:
            from netCDF4 import Dataset
            topo = './GMRTv4_0_20220922topo.grd'
            var2 = Dataset(topo, mode='r')
            xtopo=np.arange(var2.variables['x_range'][0],var2.variables['x_range'][1]+0.5*var2.variables['spacing'][0],var2.variables['spacing'][0])
            ytopo=np.arange(var2.variables['y_range'][0],var2.variables['y_range'][1]+0.5*var2.variables['spacing'][1],var2.variables['spacing'][1])
            z=var2.variables['z'][:]
            xt,yt=np.meshgrid(xtopo,ytopo)
            z=np.reshape(z,xt.shape)[::-1,:]
            ret = ax.projection.transform_points(ccrs.PlateCarree(), xt, yt)
            xx = ret[..., 0]
            yy = ret[..., 1]
            #ax.contour(xx, yy, z,levels=np.arange(500,5100,1000), colors='0.2', linewidths=0.3, zorder=3)
            ax.contour(xx, yy, z,levels=[250,1000,2000,3000,4000,5000], colors='0.2', linewidths=0.3, zorder=3)
        except:
            print('Could not find elevation data to plot topography')
    ## rivers:
    rios = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none' )
    ax.add_feature(rios, edgecolor='gray', linewidth=0.2 )
    # borders:
    limites_int = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
    facecolor='none')
    ax.add_feature(limites_int, edgecolor='k',linewidth=0.3, linestyle='-')

    if remove_borders:
        if loncorners==None:
            xi,yi=area.lon_corners[1:-1,1:-1],area.lat_corners[1:-1,1:-1]
        else:
            xi,yi=loncorners[0][1:-1,1:-1],latcorners[0][1:-1,1:-1]
        if not logscale:
            cs = plt.pcolormesh(xi,yi,T[1:-1,1:-1],transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,cmap=cmap )
        else:
            cs = plt.pcolormesh(xi,yi,T[1:-1,1:-1],transform=ccrs.PlateCarree(),norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap )
    else:
        if loncorners==None:
            xi,yi=area.lon_corners,area.lat_corners
        else:
            xi,yi=loncorners[0],latcorners[0]
        if not logscale:
            cs = plt.pcolormesh(xi,yi,T[:,:],transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,cmap=cmap)
        else:
            cs = plt.pcolormesh(xi,yi,T[1:-1,1:-1],transform=ccrs.PlateCarree(),norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap )
    if title==None:
        plt.title('%d-%02d-%02d-%02d:%02d'%(time[0],time[1],time[2],time[3],time[4]))
    else:
        plt.title(title,fontsize=16)
    if cb:
        if extend=='neither':
            cax=ax.inset_axes([0.025, -0.09,0.95, 0.03])
        else:
            cax = ax.inset_axes([0.05, -0.09, 0.9, 0.03])
        if not logscale:
            cbar = plt.colorbar(cs,extend=extend, pad=0., ticks=ticks, orientation='horizontal',cax=cax)
            cbar.set_label(label=label,size=12)
            cbar.ax.tick_params(labelsize=12)
        else:
            cbar = plt.colorbar(cs,extend=extend, pad=0.15, ticks=ticks, orientation='horizontal',cax=cax)
            cbar.set_label(label=label,size=12)
            cbar.ax.tick_params(labelsize=12)
    if not(np.all(np.isnan(corners))):
        x=[corners[3],corners[1],corners[1],corners[3],corners[3]]
        y=[corners[0],corners[0],corners[2],corners[2],corners[0]]
        x1,y1=mp(x,y)
        plt.plot(x1,y1,lw=0.5,c='b',zorder=11)
    if boxes!=None:
        for i in range(len(boxes)):
            if len(boxescolor)==1:
                plt.plot([boxes[i][0],boxes[i][0],boxes[i][1],boxes[i][1],boxes[i][0]],[boxes[i][2],boxes[i][3],boxes[i][3],boxes[i][2],boxes[i][2]],lw=1,color=boxescolor[0],transform=ccrs.PlateCarree())
                plt.annotate('%d'%(i+1),xy=(boxes[i][0]+0.1,boxes[i][2]+0.1),fontsize=12,fontstyle='italic',transform=ccrs.PlateCarree(),color=boxescolor[0])
            else:
                plt.plot([boxes[i][0],boxes[i][0],boxes[i][1],boxes[i][1],boxes[i][0]],[boxes[i][2],boxes[i][3],boxes[i][3],boxes[i][2],boxes[i][2]],lw=1,color=boxescolor[i],transform=ccrs.PlateCarree())
                plt.annotate('%d'%(i+1),xy=(boxes[i][0]+0.1,boxes[i][2]+0.1),fontsize=12,fontstyle='italic',transform=ccrs.PlateCarree(),color=boxescolor[i])
    return cs

def find_delta_t(time):
    # find the typical delta_t of the images by looking at the first 100
    # images and finding the most frequent time interval (in minutes)
    from scipy import stats
    t0 = dt.datetime(*time[0])
    Dt = []
    for i in range(1,100):
        Dt.append((dt.datetime(*time[i])-t0).total_seconds())
        t0 = dt.datetime(*(time[i]))
    return stats.mode(Dt)[0][0]/60



