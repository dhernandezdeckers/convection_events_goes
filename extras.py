import numpy as np

def joblib_get_hmax_TRMM(i,ny,mask,lon_corners,lat_corners,data,min_TRMM_precip,dTmin2h):
    H_max = np.ones(ny)*np.nan
    for j in range(ny-2):
        if mask[i,j+1]==1:
            H_max[j+1] = get_max_hour_TRMM(i,j+1,lon_corners,lat_corners,data,min_TRMM_precip,dTmin2h)
    return H_max 

def joblib_get_hmax(i,ny,mask,lon_corners,lat_corners,data,dTmin2h):
    H_max = np.ones(ny)*np.nan
    for j in range(ny-2):
        if mask[i,j+1]==1:
            H_max[j+1] = get_max_hour(i,j+1,lon_corners,lat_corners,data,dTmin2h)
    return H_max 

def get_max_hour_TRMM(i,j,lon_corners,lat_corners,data,min_TRMM_precip,dTmin2h):
    histogram=np.ones(24)*np.nan
    for k in range(24):
        ind = np.where((data[:,6]==k)*(data[:,1]>=lon_corners[i-1,j])*(data[:,1]<=lon_corners[i+2,j])*(data[:,2]<=lat_corners[i,j+2])*(data[:,2]>=lat_corners[i,j-1])*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
        histogram[k]=len(ind)
    return np.where(histogram==np.nanmax(histogram))[0][0]#, S1

def get_max_hour(i,j,lon_corners,lat_corners,data,dTmin2h):
    histogram=np.ones(24)*np.nan
    for k in range(24):
        ind = np.where((data[:,6]==k)*(data[:,1]>=lon_corners[i-1,j])*(data[:,1]<=lon_corners[i+2,j])*(data[:,2]<=lat_corners[i,j+2])*(data[:,2]>=lat_corners[i,j-1])*(data[:,8]<=dTmin2h))[0]
        histogram[k]=len(ind)
    return np.where(histogram==np.nanmax(histogram))[0][0]#, S1

def get_N_events_h(data,nx,ny,min_TRMM_precip,dTmin2h,hh,mask):
    N_events_h=np.zeros([nx,ny])
    ind=np.where((data[:,6]==hh)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
    for j in range(len(ind)):
        case=data[ind[j]]
        ind_lon=np.where(np.round(area.lon_centers,3)==case[1])[0][0]
        ind_lat=np.where(np.round(area.lat_centers,3)==case[2])[1][0]
        if mask[ind_lon,ind_lat]==1:
            N_events_h[ind_lon,ind_lat]+=1
    histogram = np.int(np.sum(N_events_h))
    return histogram, N_events_h

def get_N_events_m(data,nx,ny,min_TRMM_precip,dTmin2h,mm,mask):
    N_events_m=np.zeros([nx,ny])
    ind=np.where((data[:,4]==mm+1)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0] #only consider cases with TRMM precip above threshold and minimum decrease in CTT of dTmin2h in 2h
    for j in range(len(ind)):
        case=data[ind[j]]
        ind_lon=np.where(np.round(area.lon_centers,3)==case[1])[0][0]
        ind_lat=np.where(np.round(area.lat_centers,3)==case[2])[1][0]
        if mask[ind_lon,ind_lat]==1:
            N_events_m[ind_lon,ind_lat]+=1
    histogram_m = np.int(np.sum(N_events_m))
    return histogram_m, N_events_m

 
