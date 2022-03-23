import numpy as np
from netCDF4 import Dataset
import GOES_grid as grid
from GOES_grid import *
import os
import pickle
import pdb
from joblib import Parallel, delayed
import datetime as dt

"""
This script is the first step for convective event identification based on IR data
as presented in:
    Hernandez-Deckers, D.. (2022) Features of atmospheric deep convection
    in northwestern South America obtained from infrared satellite data.
    Q J R Meteorol Soc, 148( 742), 338– 350. https://doi.org/10.1002/qj.4208

This first step reads brightness temperature data from GOES-13 infrarred band (10.7um),
and averages it to a uniform lat-lon grid defined by the user. It has been tested 
using netcdf files obtained from:
NOAA (1994) Geostationary Operational Environmental Satellite (GOES) imager data. 
GVAR_IMG band 4, NOAA National Centers for Environmental Information, Office of 
Satellite and Product Operations. 
https://www.avl.class.noaa.gov/saa/products/search?datatype_family=GVAR_IMG

After running this script, the script "find_convective_events.py" uses this result 
to identify and track convective events and obtain spatio-temporal statistics of them.
All output from these scripts will be saved to a folder with name "CONVECTION_*'
where the user defines the ending (case_name).

Both codes are free to use with no warranty. For reference to this method, please cite:

    Hernandez-Deckers, D.. (2022) Features of atmospheric deep convection
    in northwestern South America obtained from infrared satellite data.
    Q J R Meteorol Soc, 148( 742), 338– 350. https://doi.org/10.1002/qj.4208

D. Hernández-Deckers - 2022
dhernandezd@unal.edu.co
"""

# ************************************************************************************
# Main user settings:
case_name   = 'test'                # optional, for file names. Can also be left blanck ('')
path        = '/media/Drive/GOES/'  # path to GOES images (netcdf format)
n_jobs      = 47                    # Number of jobs for parallelization (uses joblib)
Ea_r        = 6378                  # Earth radius to compute distances from lat lon coordinates
UTC         = -5                    # Conversion from UTC to local time
t00         = dt.date(2011,1,1)     # Starting date in datetime format
tff         = dt.date(2017,12,31)   # Final date in datetime format

"""
NOTE:
GOES images should be stored in "path", each year in one folder, each month in one folder.
For example: path+'/2011/01/goes13.YYYY.DDD.*.nc'
(where DDD is day of year)
"""

# ************************************************************************************
# Parameters for defining the study area. Since it is a 'rectangular' lat lon grid, 
# only the grid size and the edge's latitudes and longitudes are required:
nx      = 80                        # number of gridcells in x
ny      = 106                       # number of gridcells in y
Slat    = -2.5                      # southern latitude
Nlat    = 12.75                     # northern latitude
Wlon    = -80                       # western longitude
Elon    = -68.5                     # eastern longitude


# ************************************************************************************
# If part of the domain wants to be masked (optional), this has to be done manually here.
mask=np.ones([nx,ny]) # this means no mask (entire grid is used)

# If mask is needed, set masked gridboxes to zero. For example:
#mask[0,:]=0
#mask[1,0]=mask[1,8:]=0
#mask[2,0]=mask[2,10:]=0
#mask[3,:5]=mask[3,14:]=0
#mask[4,:6]=mask[4,16:]=0
#mask[5,:8]=mask[5,16:]=0
#mask[6,:8]=mask[6,18]=mask[6,20:]=0
#mask[7,:9]=mask[7,-1]=0
#mask[8,:11]=mask[8,-1]=0
#mask[9,:11]=mask[9,-1]=0
#mask[10,:13]=mask[10,-2:]=0
#mask[11,:15]=mask[11,-4:]=0
#mask[12,:17]=mask[12,20:]=0
#mask[13,:]=0

# ************************************************************************************
# ***************** END OF USER PARAMETERS ********************************************
# ************************************************************************************


# ************************************************************************************
# create Grid object:
area=grid.Grid( Slat, Elon, Nlat, Wlon, nx=nx, ny=ny, ER=Ea_r, UTC=UTC )
area.create_mask(mask)

print("Will now create a folder named 'CONVECTION_"+case_name+"' where all files will be saved.\n")
folder='CONVECTION_'+case_name
if not os.path.exists(folder):
    os.makedirs(folder)

# ************************************************************************************
# Grid object is saved to a file (for later use in other scripts)
pickle.dump( area, open(folder+'/area_nxny%d%d.p'%(nx,ny),'wb'))

# ************************************************************************************
# read in chunks of XX days
days_per_chunk=15
t=t00
T_grid = []
time = []
print('Reading files:')
while t <= tff:
    counter=0
    lons=[]
    lats=[]
    img =[]
    date=[]
    times=[]
    while t<=tff and counter<days_per_chunk:
        ls_list=os.popen('ls '+ path + '%04d/%02d/goes13.%04d.%03d.'%(t.year,t.month,t.year,t.timetuple().tm_yday) + '*.nc').read().split() #one day
        for ifile in ls_list:
            try:
                var=Dataset(ifile, mode='r')
                lons.append(var.variables['lon'][:])
                lats.append(var.variables['lat'][:])
                img.append(var.variables['data'][:])
                date.append(var.variables['imageDate'][:])
                times.append(var.variables['imageTime'][:])
                print(ifile, end='\r')
                var.close()
                var=None
            except:
                print('could not read file '+ifile)
        t+=dt.timedelta(days=1)
        counter+=1
    print('\nProcessing %d days of data with %d jobs...'%(days_per_chunk,n_jobs))
    if (int(len(lons)/n_jobs)-len(lons)/n_jobs) !=0:
        len_job=int(len(lons)/n_jobs)+1
    else: 
        len_job=int(len(lons)/n_jobs)
    jobs=[]
    for i in range(n_jobs-1):
        jobs.append([area,lons[i*len_job:(i+1)*len_job],lats[i*len_job:(i+1)*len_job],img[i*len_job:(i+1)*len_job],date[i*len_job:(i+1)*len_job],times[i*len_job:(i+1)*len_job]])
    i=n_jobs-1
    jobs.append([area,lons[i*len_job:],lats[i*len_job:],img[i*len_job:],date[i*len_job:],times[i*len_job:]])
    out=Parallel(n_jobs=n_jobs)(delayed(read_job)(*jobs[i]) for i in range(len(jobs)))
    for i in range(n_jobs):
        T_grid.extend(out[i][0])
        time.extend(out[i][1])
T_grid=np.asarray(T_grid)
time=np.asarray(time)

# ************************************************************************************
# Now make sure the data is in the correct time-order (safety check/fix):
T_grid2=[]
time2=[]
T_grid2.append(T_grid[0])
time2.append(time[0])
i=1
while i<T_grid.shape[0]:
    deltat = (dt.datetime(*time[i])-dt.datetime(*time[i-1])).total_seconds()/60
    if deltat<0:
        i0=i-1
        while dt.datetime(*time[i])<=dt.datetime(*time[i0]):
            print(i)
            i+=1
    T_grid2.append(T_grid[i])
    time2.append(time[i])
    i+=1
T_grid2=np.asarray(T_grid2)
time2=np.asarray(time2)

np.save(folder+'/T_grid_nxny%d%d.npy'%(nx,ny),T_grid2)  # THIS ARRAY CONTAINS THE GRIDDED BT
np.save(folder+'/time_nxny%d%d.npy'%(nx,ny),time2)      # THIS ARRAY CONTAINS THE TIME INFORMATION OF THE GRIDDED BT
