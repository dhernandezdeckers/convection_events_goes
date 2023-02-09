import numpy as np
import s3fs
import calendar
import os
import pdb
from extras import *
from joblib import Parallel, delayed

# initial date:
yy0 = 2022
mm0 = 12 
dd0 = 1

# final date:
yyf = 2022
mmf = 12
ddf = 31

n_jobs = 31

crop = True

# latlon box to crop: (only if crop==True)
n_lat = 25
s_lat = -15
w_lon = -90
e_lon = -50

path = '/media/HD3/GOES16'

REMOVE_ORIGINAL_FILES = True # (only used if crop is True)


#**********************************************************************************

header = 'OR_ABI-L1b-RadF-'#M3C13_G16_

sdate = dt.datetime(yy0,mm0,dd0)
edate = dt.datetime(yyf,mmf,ddf)

def get_crop_GOESR_period(sdate,edate,s_lat,n_lat,w_lon,e_lon):
    # Use the anonymous credentials to access public data
    fs = s3fs.S3FileSystem(anon=True)
    
    # List contents of GOES-16 bucket.
    fs.ls('s3://noaa-goes16/')
    
    # List specific files of GOES-17 CONUS data (multiband format) on a certain hour
    # Note: the `s3://` is not required
    idate = sdate
    iyear = idate.year
    
    existing_files = os.listdir(path+'/%04d/'%(iyear)) # to avoid downloading a previously downloaded file
    counter = 0
    while idate<=edate:
        if idate.year!=iyear:
            iyear=idate.year
            existing_files = os.listdir(path+'/%04d/'%(iyear)) # to avoid downloading a previously downloaded file
        iday = idate.timetuple().tm_yday
        for hh in range(24):
            files = np.array(fs.ls('noaa-goes16/ABI-L1b-RadF/%04d/%03d/%02d/'%(iyear,iday,hh)))
            for ifile in files:
                fname = ifile.split('/')[-1]
                if fname in existing_files or fname[:-3]+'_COL.nc' in existing_files:
                    print(ifile + ' already exists. Will skip download!')
                else:
                    if fname[18:21]=='C13':
                        try:
                            # DOWNLOAD ORIGINAL FILE:
                            fs.get(ifile,path+'/%04d/'%(iyear)+fname)
                            print('Downloaded '+ifile)
                        except:
                            print('Could not download '+ifile)
                if crop and fname[18:21]=='C13':
                    # CROP REGION:
                    newfile_name = fname[:-3]+'_COL.nc'
                    if newfile_name in existing_files:
                        print(newfile_name+' already exists. Will skip!')
                    else:
                        or_file = path+'/%04d/'%(iyear)+fname
                        print('processing '+or_file)
                        newfile = path+'/%04d/'%(iyear)+newfile_name
                        success = crop_GOES16_file(or_file, newfile, s_lat, n_lat, w_lon, e_lon)
                        #success = compute_BT_and_crop(or_file, newfile, s_lat, n_lat, w_lon, e_lon, fillvalue, crop)
                        if success:
                            print('cropped successfully '+fname)
                            if REMOVE_ORIGINAL_FILES:
                                os.remove(or_file)
                        else:
                            print('Could not process '+fname)
        idate=idate+dt.timedelta(days=1)
        counter+=1
    return counter

jobs=[]
if np.mod((edate-sdate).days,n_jobs)==0:
    days_per_job = int((edate-sdate).days/n_jobs)
else:
    days_per_job = int((edate-sdate).days/n_jobs) + 1

for i in range(n_jobs):
    fdate = np.min([sdate+dt.timedelta(days=days_per_job-1),edate])
    jobs.append( (sdate,fdate,s_lat,n_lat,w_lon,e_lon) )
    sdate=fdate+dt.timedelta(days=1)

out = Parallel(n_jobs=n_jobs)(delayed(get_crop_GOESR_period)(*jobs[i]) for i in range(len(jobs)))
print('Finished downloading %d days of data'%(np.sum(out)))


