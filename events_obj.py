import numpy as np
from netCDF4 import Dataset
import datetime as dt
import os

class Event(object):

    def __init__( self, t0, tf, coords, peaks, dx, dy, TRMM=False, TRMM_data_path='./', UTC_offset=5):
        self.t0=t0
        self.t=[t0]
        self.tf=tf
        self.tf_hrs = (dt.datetime(*self.tf)-dt.datetime(2010,1,1)).total_seconds()/3600. # hours since 1.1.2010 to tf
        self.coords=[]
        self.coords.append(coords)
        self.peaks=[]
        self.peaks.append(peaks)
        self.area=dx*dy
        self.add_dT2h(0,None,None)
        self.TRMM_precip=[]
        if TRMM: # if 3-hourly TRMM precip is used to verify event
            self.TRMM_precip.append(find_TRMM_val(dt.datetime(*self.t0)+dt.timedelta(hours=UTC_offset),self.peaks[-1][0],self.peaks[-1][1],TRMM_data_path))
        else: # if not, assign 0 (this condition will be ignored anyway)
            self.TRMM_precip.append(0)

    def add_system( self, event): # this adds one timestep to an event
        self.t.append(event.tf)
        self.tf = event.tf
        self.tf_hrs = (dt.datetime(*self.tf)-dt.datetime(2010,1,1)).total_seconds()/3600.
        self.coords.append(event.coords[-1])
        self.peaks.append(event.peaks[-1])
        self.TRMM_precip.append(event.TRMM_precip[-1])

    def find_minT(self):
        peaks=np.asarray(self.peaks)
        ind=np.where(peaks[:,2]==np.min(peaks[:,2]))[0][0]
        self.minT_ind=ind
        self.minT=self.t[ind]

    def add_dT2h(self, dT,dT_t,dT_coord):
        self.dT=dT
        self.dT_t=dT_t
        self.dT_coord=dT_coord

    def find_event_vel(self, Ea_r):
        """
        find the velocity of the system in between each pair of timesteps (assigned to the first point, so will have one timestep less)
        based on a centroid of the 235K region (approximate with planar centroid)
        and the mean velocity throughout time
        """
        import pdb
        self.uv = np.ones((len(self.t)-1,4))*np.nan
        x_c = np.zeros(len(self.t))
        y_c = np.zeros(len(self.t))
        for i in range(len(self.t)): # compute centroid at each time step
            centroids= np.mean(np.asarray(self.coords[i]),axis=0)
            x_c[i] = centroids[0]
            y_c[i] = centroids[1]
        for i in range(len(self.t)-1):
            dy = Ea_r*(y_c[i+1]-y_c[i])*np.pi/180
            dx = Ea_r*np.cos(0.5*(y_c[i+1]+y_c[i])*np.pi/180)*(x_c[i+1]-x_c[i])*np.pi/180
            dtime = (dt.datetime(*self.t[i+1])-dt.datetime(*self.t[i])).total_seconds()/3600 # in hours
            self.uv[i,:] = [x_c[i], y_c[i], dx/dtime, dy/dtime]
        if len(self.t)>1:
            self.uvmean = np.mean(self.uv,axis=0) # mean values: lon, lat, u, v
        else:
            self.uvmean = [x_c[0], y_c[0], 0, 0]

def find_TRMM_val(t_in,lon,lat,path= '/media/Drive/TRMM_3HR/netcdf/'):
    """
    For each event, find the 3hr precipitation rate (in mm/hr) from TRMM
    (the maximum value in a 3x3 grid centered at lon lat): TRMM data is 
    every 3 hours, with timestamp centered in those three hours (UTC)
    first find the TRMM data file that corresponds to the time in t_in:
    """
    TRMM_times=np.arange(0,25,3)
    T=np.where(np.abs(TRMM_times-(t_in.hour+t_in.minute/60.))<=1.5)[0][0]
    T_TRMM=t_in.replace(hour=0,minute=0)+dt.timedelta(hours=int(TRMM_times[T]))
    fname=os.popen('ls '+path+'3B42.%4d%02d%02d.%02d* '%(T_TRMM.year,T_TRMM.month,T_TRMM.day,T_TRMM.hour)).read().strip()
    dlat=0.25
    dlon=0.25
    try:
        ncfile = Dataset(fname,mode='r')
        # now get the data value at the corresponding lon lat location
        data = ncfile.variables['precipitation'][:]
        data_lon = ncfile.variables['nlon'][:]
        data_lat = ncfile.variables['nlat'][:]
        lon_corners = data_lon-dlon*0.5
        lat_corners = data_lat-dlat*0.5
        llon=np.where(lon_corners<lon)[0][-1]
        llat=np.where(lat_corners<lat)[0][-1]
        # find the maximum value of precipitation in the surrounding gridboxes:
        lon1=np.max([0,llon-1])
        lon2=np.min([data.shape[0],llon+2])
        lat1=np.max([0,llat-1])
        lat2=np.min([data.shape[1],llat+2])
        return np.max(data[lon1:lon2,lat1:lat2])
    except:
        print('file '+fname+' not found!')
        return np.nan



