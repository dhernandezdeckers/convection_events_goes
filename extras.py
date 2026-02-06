import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pdb

def read_namelist_parameters(fname='namelist.txt'):
    """
    reads all main parameters for event identification and tracking
    and returns them in a dictionary
    """
    namelist = open(fname,'r')
    lines = namelist.readlines()

    i=0
    params = {}
    while i<len(lines):
        if not(lines[i][0] in ['#','\n',' ']):
            line=(lines[i].split('#')[0]).replace(' ','').strip().split('=')
            if len(line)==2:
               params[line[0]]=line[1] 
        i+=1
    return params

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

def get_N_events_h(data,nx,ny,min_TRMM_precip,dTmin2h,hh,mask,lon_centers,lat_centers):
    N_events_h=np.zeros([nx,ny])
    ind=np.where((data[:,6]==hh)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
    for j in range(len(ind)):
        case=data[ind[j]]
        ind_lon=np.where(np.round(lon_centers,3)==case[1])[0][0]
        ind_lat=np.where(np.round(lat_centers,3)==case[2])[1][0]
        if mask[ind_lon,ind_lat]==1:
            N_events_h[ind_lon,ind_lat]+=1
    histogram = int(np.sum(N_events_h))
    return histogram, N_events_h

def get_N_events_m(data,nx,ny,min_TRMM_precip,dTmin2h,mm,mask,lon_centers,lat_centers):
    N_events_m=np.zeros([nx,ny])
    ind=np.where((data[:,4]==mm+1)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0] #only consider cases with TRMM precip above threshold and minimum decrease in CTT of dTmin2h in 2h
    for j in range(len(ind)):
        case=data[ind[j]]
        ind_lon=np.where(np.round(lon_centers,3)==case[1])[0][0]
        ind_lat=np.where(np.round(lat_centers,3)==case[2])[1][0]
        if mask[ind_lon,ind_lat]==1:
            N_events_m[ind_lon,ind_lat]+=1
    histogram_m = int(np.sum(N_events_m))
    return histogram_m, N_events_m

def moving_avg(arr,window=3):
    if np.mod(window,2)==0:
        window+=1
        print('will use window of %d instead (must be odd number)'%(window))
    out_arr=np.ones_like(arr)*np.nan
    for i in range(len(arr)):
        if (i-int((window-1)*0.5))<0:
            out_arr[i] = np.nanmean(np.append(arr[i-int((window-1)*0.5):],arr[:i+int((window-1)*0.5)+1]))
        elif i+int((window-1)*0.5)+1>len(arr):
            out_arr[i] = np.nanmean(np.append(arr[i-int((window-1)*0.5):],arr[:i+int((window-1)*0.5)+1-len(arr)]))
        else:
            out_arr[i] = np.nanmean(arr[i-int((window-1)*0.5):i+int((window-1)*0.5)+1])
    return out_arr

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_histograms_box_hourly(data,slat,nlat,wlon,elon,min_TRMM_precip,dTmin2h):
    histogram_h=np.zeros(24)
    for hh in range(24):
        ind=np.where((data[:,6]==hh)*(data[:,1]>=wlon)*(data[:,1]<=elon)*(data[:,2]<=nlat)*(data[:,2]>=slat)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
        if len(ind)>0:
            histogram_h[hh]=len(ind)
    histogram_h_smooth=moving_avg(histogram_h)
    hour_max = np.where(histogram_h_smooth==np.nanmax(histogram_h_smooth))[0]
    return histogram_h, hour_max[0]

def get_histograms_box(data,slat,nlat,wlon,elon,min_TRMM_precip,dTmin2h):
    histogram_m=np.zeros(12)
    for mm in range(12):
        ind=np.where((data[:,4]==mm+1)*(data[:,1]>=wlon)*(data[:,1]<=elon)*(data[:,2]<=nlat)*(data[:,2]>=slat)*(data[:,16]>=min_TRMM_precip)*(data[:,8]<=dTmin2h))[0]
        if len(ind)>0:
            histogram_m[mm]=len(ind)
    histogram_m_smooth=moving_avg(histogram_m)
    month_max = np.where(histogram_m_smooth==np.nanmax(histogram_m_smooth))[0]
    return histogram_m, month_max[0]

def datetime_from_secondsepoch( sec, epoch=dt.datetime(2000,1,1,12,00)):
    return epoch + dt.timedelta(seconds=sec)

def compute_latlon_GOES16( datafile ):
    # compute lat lon geodetic coordinates from N/S Elevation angle and E/W scanning angle (x), 
    # following the GOES-R SERIES PRODUCT DEFINITION AND USERS' GUIDE (VOL 3) 
    # (https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf, p.21)
    #******************************************************************************************
    radx   = datafile.variables['x'][:]
    rady   = datafile.variables['y'][:]
    rad_x, rad_y = np.meshgrid(radx,rady)

    GIP = datafile.variables['goes_imager_projection']
    H   = GIP.perspective_point_height + GIP.semi_major_axis
    req = GIP.semi_major_axis
    rpol= GIP.semi_minor_axis
    a   = np.sin(rad_x)**2+(np.cos(rad_x)**2)*(np.cos(rad_y)**2+(np.sin(rad_y)**2)*(req**2/(rpol**2)))
    b   = -2*H*np.cos(rad_x)*np.cos(rad_y)
    c   = H*H-req*req
    mask= (b*b>=4*a*c)
    rs  = np.empty_like(b)
    #rs = (-b - np.sqrt((b**2-4*a*c)))/(2*a)
    rs[mask] = (-b[mask]-np.sqrt((b**2-4*a*c)[mask]))/(2*a[mask])
    rs[~mask] = np.nan
    Sx  = rs*np.cos(rad_x)*np.cos(rad_y)
    Sy  = -rs*np.sin(rad_x)
    Sz  = rs*np.cos(rad_x)*np.sin(rad_y)
    lat = np.arctan((req**2/rpol**2)*Sz/np.sqrt((H-Sx)**2+Sy**2))
    L0  = GIP.longitude_of_projection_origin*np.pi/180
    lon = L0-np.arctan(Sy/(H-Sx))
    return lat*180/np.pi, lon*180/np.pi

def compute_elev_scan_angle_GOES16( datafile, lat, lon ):
    # compute elevation and scanning angle corresponding to a latitutde longitude coordinate (in deg)
    # following the GOES-R SERIES PRODUCT DEFINITION AND USERS' GUIDE (VOL 3) 
    # (https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf, p.21)
    #******************************************************************************************
    radx   = datafile.variables['x'][:]
    rady   = datafile.variables['y'][:]
    rad_x, rad_y = np.meshgrid(radx,rady)

    phi = lat*np.pi/180 # latitude in radians
    lda = lon*np.pi/180 # longitude in radians
    GIP = datafile.variables['goes_imager_projection']
    H   = GIP.perspective_point_height + GIP.semi_major_axis
    req = GIP.semi_major_axis
    rpol= GIP.semi_minor_axis
    L0  = GIP.longitude_of_projection_origin*np.pi/180
    e = 0.0818191910435
    phi_c = np.arctan((rpol**2/req**2)*np.tan(phi))
    r_c = rpol/(np.sqrt(1-e**2*np.cos(phi_c)**2))
    Sx = H-r_c*np.cos(phi_c)*np.cos(lda-L0)
    Sy = -r_c*np.cos(phi_c)*np.sin(lda-L0)
    Sz = r_c*np.sin(phi_c)
    y = np.arctan(Sz/Sx)
    x = np.arcsin(-Sy/(np.sqrt(Sx**2+Sy**2+Sz**2)))
    return x, y


def compute_BT( fk1,fk2,bc1,bc2,Lv ):
    # Brightness temperature from radiances, following the GOES-R SERIES PRODUCT DEFINITION 
    # AND USERS' GUIDE (VOL 3) (https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf, p.28)
    #*****************************************************************************************
    c1 = (fk1/Lv)+1
    c1[np.where(c1<=0)]=np.ma.masked
    #return (fk2/np.log((fk1/Lv)+1)-bc1)/bc2
    return (fk2/np.ma.log(c1)-bc1)/bc2


def crop_GOES16_file(or_fname, new_fname, s_lat, n_lat, w_lon, e_lon):
    #***********************************************************************
    # reads a GOES-16 file with radiances, crops a lat-lon box and saves it
    # in a new file.
    #***********************************************************************
    try:
        idata = Dataset(or_fname, mode='r')
        odata = Dataset(new_fname, mode='w',format='NETCDF4')
        x_ll, y_ll = compute_elev_scan_angle_GOES16(idata, s_lat, w_lon)
        x_ul, y_ul = compute_elev_scan_angle_GOES16(idata, n_lat, w_lon)
        x_ur, y_ur = compute_elev_scan_angle_GOES16(idata, n_lat, e_lon)
        x_lr, y_lr = compute_elev_scan_angle_GOES16(idata, s_lat, e_lon)
        xmin = np.min([x_ll, x_ul])
        xmax = np.max([x_ur, x_lr])
        ymin = np.min([y_ll, y_lr])
        ymax = np.min([y_ul, y_ur])
        radx = idata.variables['x'][:]
        rady   = idata.variables['y'][:]
        xind = np.where((radx>=xmin)*(radx<=xmax))
        yind = np.where((rady>=ymin)*(rady<=ymax))
        x0 = np.min(xind)
        x1 = np.max(xind)
        y0 = np.min(yind)
        y1 = np.max(yind)
        rad = idata.variables['Rad'][y0:y1,x0:x1]
        DQF = idata.variables['DQF'][y0:y1,x0:x1]
        x_r = idata.variables['x'][x0:x1]
        y_r = idata.variables['y'][y0:y1]
        fk1 = idata.variables['planck_fk1'][:]
        fk2 = idata.variables['planck_fk2'][:]
        bc1 = idata.variables['planck_bc1'][:]
        bc2 = idata.variables['planck_bc2'][:]

        #create new netcdf file

        for dname, the_dim in idata.dimensions.items():
            if dname=='y':
                odata.createDimension(dname, len(y_r) if not the_dim.isunlimited() else None)
            elif dname=='x':
                odata.createDimension(dname, len(x_r) if not the_dim.isunlimited() else None)
            else:
                odata.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
            #print( odata.dimensions[dname])

        copy_vars = ['x','y','Rad','DQF','t','planck_fk1','planck_fk2','planck_bc1','planck_bc2','goes_imager_projection']
        crop_vars = ['x','y','Rad','DQF']
        for name, variable in idata.variables.items():
            if name in copy_vars:
                x = odata.createVariable(name, variable.datatype, variable.dimensions, zlib=True)
                x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                if not (name in crop_vars):
                    odata.variables[name][:] = idata.variables[name][:]
        #odata.variables['x'][:] = (np.rint((x_r-idata.variables['x'].add_offset)/idata.variables['x'].scale_factor)).astype(int)    
        #odata.variables['y'][:] = (np.rint((y_r-idata.variables['y'].add_offset)/idata.variables['y'].scale_factor)).astype(int)
        #odata.variables['Rad'][:,:] = (np.rint((rad-idata.variables['Rad'].add_offset)/idata.variables['Rad'].scale_factor)).astype(int)
        odata.variables['x'][:] = x_r 
        odata.variables['y'][:] = y_r
        odata.variables['Rad'][:,:] = rad
        odata.variables['DQF'][:,:] = DQF

        odata.close()
        idata.close()
        success = True
    except:
        print('******** Could not crop file '+or_fname)
        success = False
    return success


def compute_latlon_BT(or_fname):
    #**************************************************************************************
    # reads a file with GOES-16 ch13 radiances in elevation and scanning angle coordinates
    # and returns the netCDF Dataset, and the corresponding lat lon grid in degrees and 
    # brightess temperatures in K
    #**************************************************************************************
    try:
        idata = Dataset(or_fname, mode='r')
        lats,lons = compute_latlon_GOES16(idata)
        rads = idata.variables['Rad'][:,:]
        DQFs = idata.variables['DQF'][:,:]
        
        fk1 = idata.variables['planck_fk1'][:]
        fk2 = idata.variables['planck_fk2'][:]
        bc1 = idata.variables['planck_bc1'][:]
        bc2 = idata.variables['planck_bc2'][:]
        
        # compute brightness temperatures for the cropped area:
        BT = compute_BT( fk1, fk2, bc1, bc2, rads )
        BT[np.where(DQFs!=0)] = np.nan  # only consider quality points!
        
        lat  = np.flipud(lats)
        lon  = np.flipud(lons)
        bt   = np.flipud(BT)
    except:
        print('******** Could not process file '+or_fname)
    return idata, lat, lon, bt


def compute_BT_and_crop(or_fname, new_fname, s_lat, n_lat, w_lon, e_lon, fillvalue, crop=True):
    # old version, not used anymore
    try:
        idata = Dataset(or_fname, mode='r')
        odata = Dataset(new_fname, mode='w',format='NETCDF4')
        lat,lon = compute_latlon_GOES16(idata)
        rad = idata.variables['Rad'][:]
        DQF = idata.variables['DQF'][:]
        if crop:
            c_lat=(s_lat+n_lat)*0.5
            c_lon=(w_lon+e_lon)*0.5
            c_lat_ind = np.where(lat<=c_lat)[0][0] #lat index of center of domain to crop
            c_lon_ind = np.where(lon<=c_lon)[1][-1]
            top_index = np.where(lat[:,c_lon_ind]>n_lat)[0][-1]
            bottom_index = np.where(lat[:,c_lon_ind]<s_lat)[0][0]
            left_index = np.where(lon[c_lat_ind,:]<w_lon)[0][-1]
            right_index = np.where(lon[c_lat_ind,:]>e_lon)[0][0]
            #lat.mask[outside]=True
            #lon.mask[outside]=True
            #rad.mask[outside]=True
            lats = lat[top_index:bottom_index+1,left_index:right_index+1]
            lons = lon[top_index:bottom_index+1,left_index:right_index+1]
            rads = rad[top_index:bottom_index+1,left_index:right_index+1]
            #rads = idata.variables['Rad'][inside]
            DQFs = DQF[top_index:bottom_index+1,left_index:right_index+1]
        else:
            lats = lat
            lons = lon
            rads = idata.variables['Rad'][:,:]
            DQFs = idata.variables['DQF'][:,:]
        
        fk1 = idata.variables['planck_fk1'][:]
        fk2 = idata.variables['planck_fk2'][:]
        bc1 = idata.variables['planck_bc1'][:]
        bc2 = idata.variables['planck_bc2'][:]
        
        # compute brightness temperatures for the cropped area:
        BT = compute_BT( fk1, fk2, bc1, bc2, rads )
        BT[np.where(DQFs!=0)] = fillvalue  # only consider quality points!
        
        #create new netcdf file 
        lat_dim  = odata.createDimension('lat', lats.shape[0]) # latitude axis
        lon_dim  = odata.createDimension('lon', lons.shape[1]) # longitude axis
        lat = odata.createVariable('lat', np.float32, ('lat','lon',), zlib=True)
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lon = odata.createVariable('lon', np.float32, ('lat','lon',), zlib=True)
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        bt = odata.createVariable('bt', np.float32, ('lat','lon'), fill_value=fillvalue, zlib=True)
        bt.units = 'K'
        bt.long_name = 'brightness_temperature'
        
        copy_vars = ['t','planck_fk1','planck_fk2','planck_bc1','planck_bc2']
        for name, variable in idata.variables.items():
            if name in copy_vars:
                x = odata.createVariable(name, variable.datatype, variable.dimensions)
                odata.variables[name][:] = idata.variables[name][:]
                odata.variables[name].long_name = idata.variables[name].long_name
                odata.variables[name].units = idata.variables[name].units
                if hasattr(idata.variables[name],'coordinates'):
                    odata.variables[name].coordinates = idata.variables[name].coordinates
        
        lat[:,:]  = np.flipud(lats)
        lon[:,:]  = np.flipud(lons)
        bt[:,:] = np.flipud(BT)
        odata.close()
        idata.close()
        success = True
    except:
        print('******** Could not process file '+or_fname)
        success = False
    return success
 
