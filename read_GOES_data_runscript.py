import numpy as np
from netCDF4 import Dataset
import GOES_grid as grid
from GOES_grid import *
from extras import *
import os
import pickle
import pdb
from joblib import Parallel, delayed
import datetime as dt
import gc
import matplotlib
matplotlib.use('Agg')

"""
This script is the first step for convective event identification based on IR data
as presented in:
    Hernandez-Deckers, D.. (2022) Features of atmospheric deep convection
    in northwestern South America obtained from infrared satellite data.
    Q J R Meteorol Soc, 148( 742), 338– 350. https://doi.org/10.1002/qj.4208

This first step reads brightness temperature data from GOES-13 infrarred band (10.7um)
or from GOES-16 "clean" infrarred band (10.3um), and averages it to a uniform lat-lon 
grid defined by the user. It has been tested using GOES-13 netcdf files obtained from:
NOAA (1994) Geostationary Operational Environmental Satellite (GOES) imager data. 
GVAR_IMG band 4, NOAA National Centers for Environmental Information, Office of 
Satellite and Product Operations. 
https://www.avl.class.noaa.gov/saa/products/search?datatype_family=GVAR_IMG
It has been tested also with GOES-16 ABI images downloaded from aws and processed following
documentation provided by https://www.goes-r.gov/resources/docs.html.

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
case_name       = 'GOES16_2018-2022_HR'    # optional, for file names. Can also be left blanck ('')
path            = '/media/HD3/GOES16/'  # path to GOES images (netcdf format)
n_jobs          = 47                    # Number of jobs for parallelization (uses joblib)
Ea_r            = 6378                  # Earth radius to compute distances from lat lon coordinates
UTC             = -5                    # Conversion from UTC to local time
t00             = dt.date(2018,1,1)    # Starting date in datetime format
tff             = dt.date(2022,12,31)   # Final date in datetime format
GOES_ver        = '16'                  # '13' (2011-2017) or '16' (2017-)
days_per_chunk  = 10                    # entire time is splitted in this number of days (limited by available memory!)
restart_run     = True                  # if job has been killed at some point, this allows to use previously saved files (T_grid and time)

"""
NOTE:
GOES images should be stored in "path", each year in one folder, each month in one folder.
For example: path+'/2011/01/goes13.YYYY.DDD.*.nc' for GOES-13
(where DDD is day of year)
"""

# ************************************************************************************
# Parameters for defining the study area. Since it is a 'rectangular' lat lon grid, 
# only the grid size and the edge's latitudes and longitudes are required:
nx      = 160#80#66                        # number of gridcells in x
ny      = 212#106#83                       # number of gridcells in y
Slat    = -2.5#-4.93                      # southern latitude
Nlat    = 12.75#7                     # northern latitude
Wlon    = -80#-76                       # western longitude
Elon    = -68.5#-66.515                    # eastern longitude


# ************************************************************************************
# If part of the domain wants to be masked (optional), this has to be done manually here.
mask=np.ones([nx,ny]) # this means no mask (entire grid is used)

# If mask is needed, set masked gridboxes to zero. For example:
#mask[0,-1]=0
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
area=grid.Grid( Slat, Elon, Nlat, Wlon, nx=nx, ny=ny, ER=Ea_r, UTC=UTC, case_name=case_name )
area.create_mask(mask)
#*************************************************************************************
print("If it does not exist already, I will now create a folder named 'CONVECTION_"+case_name+"' where all files will be saved.\n")
folder='CONVECTION_'+case_name
if not os.path.exists(folder):
    os.makedirs(folder)

# plot the grid on a map to visualize it:
area.plot_area(lllat=Slat-0.5, urlat=Nlat+0.5,lllon=Wlon-0.5,urlon=Elon+0.5,fname=folder+'/domain.png')

# ************************************************************************************
# Grid object is saved to a file (for later use in other scripts)
pickle.dump( area, open(folder+'/area_nxny%d%d.p'%(nx,ny),'wb'))

# ************************************************************************************
# read in chunks of days_per_chunk days
print('Reading files:')

counter_files=0
if restart_run:  #find the last saved file, and start reading on the next day
    print('Will use previously saved T_grid and time files.')
    ls_list=os.popen('ls '+ folder+'/time_nxny%d%d_????.npy'%(nx,ny)).read().split()
    counter_files = int(ls_list[-1][-8:-4])
    if os.path.isfile(folder+'/T_grid_nxny%d%d_%04d.npy'%(nx,ny,counter_files)):
        time = np.load(folder+'/time_nxny%d%d_%04d.npy'%(nx,ny,counter_files))[-1]
        t00=dt.datetime(*time).date()+dt.timedelta(days=1)
        OK = True
        print('Found %d files. Will continue with file %d on '%(counter_files,counter_files+1)+ t00.strftime("%m/%d/%Y"))
        counter_files+=1
    else:
        print('******ERROR: please check that both T_grid_nxny..*.npy and time_nxny..*.npy are present, and run again!\n')
        OK = False

if OK or not(restart_run):
    t=t00
    while t <= tff:
        T_grid = []
        time = []
        counter=0
        lons=[]
        lats=[]
        img =[]
        date=[]
        times=[]
        goes_v=[] # goes version (13, 14 or 15)
        while t<=tff and counter<days_per_chunk:
            if GOES_ver=='16':
                #ls_list=os.popen('ls '+ path + '%04d/OR_ABI-L1b-BTmp-M?C13_G16_s%04d%03d'%(t.year,t.year,t.timetuple().tm_yday) + '*_BTCOL.nc').read().split() #one day
                ls_list=os.popen('ls '+ path + '%04d/OR_ABI-L1b-RadF-M?C13_G16_s%04d%03d'%(t.year,t.year,t.timetuple().tm_yday) + '*_COL.nc').read().split() #one day
            elif GOES_ver=='13':
                ls_list=os.popen('ls '+ path + '%04d/%02d/goes1?.%04d.%03d.'%(t.year,t.month,t.year,t.timetuple().tm_yday) + '*.nc').read().split() #one day
            for ifile in ls_list:
                try:
                    if GOES_ver=='16':
                        goes_v.append(ifile[-57:-55])
                        if goes_v[-1]!='16':
                            print('GOES version mismatch!!!!')
                        #img.append(var.variables['bt'][:])
                        var, lat, lon, BT = compute_latlon_BT(ifile)
                        img.append(BT)
                        lons.append(lon)
                        lats.append(lat)
                        #img.append(var.variables['Rad'][:])
                        datetime = datetime_from_secondsepoch(var.variables['t'][:].data.tolist())
                        date.append(int('%04d%03d'%(t.year,t.timetuple().tm_yday)))
                        times.append(int('%02d%02d%02d'%(datetime.hour,datetime.minute,datetime.second)))
                        print(ifile[-58:], end='\r')
                    elif GOES_ver=='13':
                        var=Dataset(ifile, mode='r')
                        goes_v.append(ifile[-29:-27])
                        if goes_v[-1]!='13':
                            print('GOES version mismatch!!!!')
                        img.append(var.variables['data'][:])
                        date.append(var.variables['imageDate'][:])
                        times.append(var.variables['imageTime'][:])
                        print(ifile[-32:], end='\r')
                        lons.append(var.variables['lon'][:])
                        lats.append(var.variables['lat'][:])
                    var.close()
                    del var
                    gc.collect()
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
            jobs.append([area,lons[i*len_job:(i+1)*len_job],lats[i*len_job:(i+1)*len_job],img[i*len_job:(i+1)*len_job],date[i*len_job:(i+1)*len_job],times[i*len_job:(i+1)*len_job],goes_v[i*len_job:(i+1)*len_job]])
        i=n_jobs-1
        jobs.append([area,lons[i*len_job:],lats[i*len_job:],img[i*len_job:],date[i*len_job:],times[i*len_job:],goes_v[i*len_job:]])
        out=Parallel(n_jobs=n_jobs)(delayed(read_job)(*jobs[i]) for i in range(len(jobs)))
        for i in range(n_jobs):
            T_grid.extend(out[i][0])
            time.extend(out[i][1])
        np.save(folder+'/T_grid_nxny%d%d_%04d.npy'%(nx,ny,counter_files),np.asarray(T_grid))
        np.save(folder+'/time_nxny%d%d_%04d.npy'%(nx,ny,counter_files),np.asarray(time))
        del jobs
        del lons
        del lats
        del img
        del times
        del date
        del T_grid
        del time
        gc.collect()
        counter_files+=1
    #T_grid=np.asarray(T_grid)
    #time=np.asarray(time)
    
    #Now merge all files:
    T_grid = []
    time = []
    for i in range(counter_files):
        T_grid.extend(np.load(folder+'/T_grid_nxny%d%d_%04d.npy'%(nx,ny,i)))
        time.extend(np.load(folder+'/time_nxny%d%d_%04d.npy'%(nx,ny,i)))
        os.system('rm '+folder+'/T_grid_nxny%d%d_%04d.npy'%(nx,ny,i))
        os.system('rm '+folder+'/time_nxny%d%d_%04d.npy'%(nx,ny,i))
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
                print(i,end='\r')
                i+=1
        T_grid2.append(T_grid[i])
        time2.append(time[i])
        i+=1
    T_grid2=np.asarray(T_grid2)
    time2=np.asarray(time2)
    
    np.save(folder+'/T_grid_nxny%d%d.npy'%(nx,ny),T_grid2)  # THIS ARRAY CONTAINS THE GRIDDED BT
    np.save(folder+'/time_nxny%d%d.npy'%(nx,ny),time2)      # THIS ARRAY CONTAINS THE TIME INFORMATION OF THE GRIDDED BT
