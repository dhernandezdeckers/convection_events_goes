import numpy as np
import pickle
import pdb
from convective_events_func import *
from extras import *
import time as time2
import datetime as dt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib.gridspec as gridspec
from pylab import *
import copy
import events_obj
#from select_cases_func import *
from scipy import stats
from scipy.ndimage import gaussian_filter
import os
from calendar import monthrange
#from mpl_toolkits.basemap import Basemap

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature


# **********************************************************
# case parameters (should match those used in read_GOES_data.py and find_convective_events.py)
# **********************************************************
case_name   = 'test'    # optional, for file names. 
nx          = 64#80        # study area grid size
ny          = 80#106
deltat      = 30        # time interval between GOES images in minutes

#**********************************************************
# Parameters for convective event identification:
#**********************************************************
# threshold for brightness temperature of broader convective event area (in K):
T_min           = 235

# threshold for brightness temperature that must be found within the broader area at least at one gridbox (in K):
T_minmin        = 210

# threshold for minimun decrease in brightness temperature in 2 hours (in K)
dTmin2h         = -50

# maximum time difference in hours between two colocated systems to be considered the same event
dt_max          = 1

# minimum 3-hourly precipitation value (in mm) according to TRMM to consider events
# (set to 0 if 3-hourly TRMM data is not used as criteria to identify events,
# set to >0 if yes):
min_TRMM_precip = 0.1

# Path where TRMM 3-hourly precipitation data (netcdf format) is located
# (only needed if min_TRMM_precip set to >0):
TRMM_data_path  = '/media/Drive/TRMM_3HR/netcdf/'

# number of jobs for parallelization:
njobs           = 48   

# conversion from local to UTC time (hours) (only used for TRMM data):
UTC_offset      = 5
#**********************************************************
#**********************************************************
print_areas = False

max_sizekm2 = 300000    # convective systems larger than this will be discarded for some analyses

# coordinate limits for plots:
lllat = -4.5
urlat = 7
lllon = -76
urlon = -66.8


boxes=[[-78.2,-77.5,3.5,4.25],
        [-78.5,-77.5,5,6],
        [-77.1,-76.4,5.5,6.5],
        [-75.4,-74.8,5,6],
        [-73.95,-73.4,7,7.65],
        [-74.7,-74,7.55,8.45],
        [-74.9,-74.1,8.55,9.2],
        [-75.5,-74.5,9.8,10.8],
        [-72.11,-71.1,9,10.3],
        [-73.1,-72.35,8.75,9.85],
        [-71,-70,6,7],
        [-73.9,-72.9,0.7,1.7] ]

#*****************************************************************************************
#********** CODE STARTS HERE **************************************************************
#*****************************************************************************************

folder='CONVECTION_'+case_name

if not os.path.exists(folder):
    print('folder '+folder+' does not exist!')
    print('Have you run the sript read_GOES_data.py already?')
    print("If yes, check 'case_name' and run again.")
else:
    T_grid  = np.load(folder+'/T_grid_nxny%d%d.npy'%(nx,ny))
    time    = np.load(folder+'/time_nxny%d%d.npy'%(nx,ny))
    area    = pickle.load( open(folder+'/area_nxny%d%d.p'%(nx,ny),'rb'),encoding='latin1')
    print('gridboxes are %.2f x %.2f km\n'%(area.dx,area.dy))

time_factor_hh, tot_time_factor = get_factor_available_data(T_grid, time, nx, ny)

np.warnings.filterwarnings('ignore')

cmap = plt.get_cmap('jet')
jet2 = truncate_colormap(cmap, 0.4, 1)

#print('4. load data_nxny%d%d_Tmin%d_T2min%d.npy and compute N_events files'%(nx,ny,T_min,T_min2))
#print('5. compute most active hours' )
#print('6. plot distribution of events per hour of day and per month of year')
#print('7. plot spatial distribution of seasonality (bimodal-unimodal)')
#print('8. write file with list of convective events')
#option= input('Enter option:\n')

#if option in ['3','6']:
N_events_total      = np.load(folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min)) 
N_events_hh         = np.load(folder+'/N_events_hh_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events_mm         = np.load(folder+'/N_events_mm_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events_total_Tmin = np.load(folder+'/N_events_total_Tmin_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min)) 
mean_ssize_Tmin     = np.load(folder+'/mean_ssize_Tmin_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min)) 
mean_sdur_Tmin      = np.load(folder+'/mean_sdur_Tmin_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events            = np.load(folder+'/N_events_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events_wTRMM      = np.load(folder+'/N_events_wTRMM_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events_wTRMM_sizelimit = np.load(folder+'/N_events_wTRMM_sizelimit_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
N_events_wTRMM_mindTpos  = np.load(folder+'/N_events_wTRMM_mindTpos_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
data                = np.load(folder+'/data_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))

mean_ssize_minBTpeak = np.load(folder+'/mean_ssize_minBTpeak_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))
mean_sdur_minBTpeak = np.load(folder+'/mean_sdur_minBTpeak_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_minmin,T_min))

ndays   = (dt.date(2018,1,1)-dt.date(2011,1,1)).days # total number of days of the time period
ndays_m = [] # number of days per month in the entire time period
for m in range(12):
    ndays_temp=0
    for y in np.arange(2011,2018):
        if m<11:
            ndays_temp+=(dt.date(y,m+2,1)-dt.date(y,m+1,1)).days
        else:
            ndays_temp+=31 #december
    ndays_m.append(ndays_temp)
ndays_m = np.asarray(ndays_m)

time_factor_hh, tot_time_factor = get_factor_available_data(T_grid, time, nx, ny)

fig=plt.figure(figsize=(14,5.9))
gs = gridspec.GridSpec(1, 4, left=0.035, right=0.99, hspace=0.2, wspace=0.05, top=0.99, bottom=0.13)
ax = subplot(gs[0],projection=ccrs.Mercator(central_longitude=-75))
cs=plot_image_cartopy(area,ax,N_events_total_Tmin[:,:]*10/(tot_time_factor*7.*area.dx*area.dy),vmin=0,vmax=10, ticks=np.arange(0,10,2), cmap='afmhot_r',label='event rate density (BT<235K)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='a)',lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
ax = subplot(gs[1],projection=ccrs.Mercator(central_longitude=-75))
N_events3_smooth = gaussian_filter(N_events_wTRMM, sigma=1, mode='reflect', cval=0.0 )
N_events3_smooth[-2,:]=np.nan
N_events3_smooth[:,1]=np.nan
cs=plot_image_cartopy(area, ax, N_events3_smooth[:,:]*10/(tot_time_factor*7.*area.dx*area.dy),vmin=0,vmax=0.275, ticks=np.arange(0,0.4,0.1), cmap='afmhot_r',label='event rate density (Tmin location)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='b)',labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
#ax = subplot(gs[2])
ax = subplot(gs[2],projection=ccrs.Mercator(central_longitude=-75))
ssize_smooth = gaussian_filter(mean_ssize_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
plot_image_cartopy(area, ax, ssize_smooth*1e-3, vmin=0, vmax=38, cmap='afmhot_r', label='mean event size (x10$^3$km$^2$))', remove_borders=True, ticks=[0,10,20,30],title='c)',logscale=False, labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
ax = subplot(gs[3],projection=ccrs.Mercator(central_longitude=-75))
sdur_smooth = gaussian_filter(mean_sdur_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
plot_image_cartopy(area, ax, sdur_smooth, vmin=0, vmax=6, cmap='afmhot_r', label='mean event duration (h)', remove_borders=True, ticks=[0,1,2,3,4,5,6],title='d)',logscale=False, labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)    
plt.savefig(folder+'/Fig1.png',dpi=300)

print('Number of events regardless of precipitation is %d'%(np.nansum(N_events)))
print('Number of events is %d'%(np.nansum(N_events_wTRMM)))
print('Number of events with size smaller than %f is %d (used for storm size and duration)'%(max_sizekm2, np.nansum(N_events_wTRMM_sizelimit)))


#if option=='5':
# ***********************************
# Find hour of most frequent events:
# ***********************************
H_max = np.ones([nx,ny])*np.nan
if min_TRMM_precip>0:
    out = Parallel(n_jobs=np.min([nx-2,njobs]))(delayed(joblib_get_hmax_TRMM)(*[i+1,ny,area.mask,area.lon_corners,area.lat_corners,data,min_TRMM_precip,dTmin2h]) for i in range(nx-2))
else:
    out = Parallel(n_jobs=np.min([nx-2,njobs]))(delayed(joblib_get_hmax)(*[i+1,ny,area.mask,area.lon_corners,area.lat_corners,data,dTmin2h]) for i in range(nx-2))

for i in range(nx-2):
    H_max[i+1,:] = out[i]

fig=plt.figure(figsize=(13.3,5.8))
gs=gridspec.GridSpec(12, 6, left=0.03, right=0.99, hspace=0.11, wspace=0.01, top=0.95, bottom=0.095, width_ratios=[1.5,1.5,0.25,1.05,0.14,1.05])#,height_ratios=[h0,h1,h0,h1,h0,h1,h0,h1,h0,h1,h0,h1])
gs0=[]
gs0.append(gs[0:2,3])
gs0.append(gs[0:2,5])
gs0.append(gs[2:4,3])
gs0.append(gs[2:4,5])
gs0.append(gs[4:6,3])
gs0.append(gs[4:6,5])
gs0.append(gs[6:8,3])
gs0.append(gs[6:8,5])
gs0.append(gs[8:10,3])
gs0.append(gs[8:10,5])
gs0.append(gs[10:12,3])
gs0.append(gs[10:12,5])
ylims=[(0,2.95),(0,5),(0,10),(0,9), (0,5), (0,16),(0,5.8), (0,22), (0,23), (0,13), (0,5.8),(0,7)]
yticks=[[1,2], [2,4], [4,8], [4,8], [2,4], [6,12], [2,4], [10,20], [10,20],[5,10], [2,4],  [3,6]]
bcolors=['w','w','k','w','w','k','w','k','w','w','k','k']
for i in range(len(boxes)):
    #j=ind_boxes[i]
    scale=1e-1
    histogram_box, hour_max = get_histograms_box_hourly(data,boxes[i][2],boxes[i][3],boxes[i][0],boxes[i][1],min_TRMM_precip,dTmin2h)
    ax=fig.add_subplot(gs0[i])
    plt.bar(np.arange(0.5,24),histogram_box*scale,width=0.75)
    plt.xticks(np.arange(0,24,3),fontsize=10)
    plt.yticks(ticks=yticks[i],fontsize=10)
    plt.xlim(0,24)
    plt.ylim(ylims[i])
    plt.annotate('%d'%(i+1),xy=(0.4,0.75),xycoords='axes fraction',fontsize=14,fontstyle='italic')
    if i<10:
        ax.xaxis.set_major_formatter(NullFormatter())
    else:
        plt.xlabel('local time (h)',fontsize=12,labelpad=1)
fig.text(0.56, 0.5, 'Number of events ($\\times10$)', va='center', rotation='vertical',fontsize=16)
fig.text(0.78, 0.97, 'c)', va='center', rotation='horizontal',fontsize=16)
ax = fig.add_subplot(gs[0:11,0],projection=ccrs.PlateCarree())
plot_image_cartopy(area, ax, H_max, vmin=0, vmax=24, ticks=np.arange(0,25,2),cmap='twilight_shifted',label='local time of max. ocurrence (h)', remove_borders=True, extend='neither',title='a)',boxes=boxes,boxescolor=bcolors,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
morning=np.where(H_max<12)
ax.scatter(area.lon_centers[morning],area.lat_centers[morning],transform=ccrs.PlateCarree(),marker='o',s=1,c='w',edgecolor='none')
ax.annotate('.......................................',xy=(0.03,-0.07),xycoords='axes fraction',zorder=10,fontsize=9,color='w')
ax.annotate('.......................................',xy=(0.03,-0.08),xycoords='axes fraction',zorder=10,fontsize=9,color='w')

LIS_file=Dataset('/media/Drive/TRMM_LIS/lis_vhrfc_1998_2013_v01.nc',mode='r')
LIS=LIS_file.variables['VHRFC_LIS_FRD'][:]
lons=np.arange(-180,180.05,0.1)
lats=np.arange(-38,38.05,0.1)
loncorners,latcorners=np.meshgrid(lons,lats)
ax = fig.add_subplot(gs[0:11,1],projection=ccrs.PlateCarree())
plot_image_cartopy(area, ax, LIS, vmin=0, vmax=150, cmap='afmhot_r', label='FRD (fl km$^{-2}$yr$^{-1}$)', remove_borders=True, title='b)', ticks=np.arange(0,151,25),loncorners=[loncorners],latcorners=[latcorners],labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)

plt.savefig(folder+'/Fig5.png',dpi=300)
#plt.savefig('hourly_events_nxny%d%d_Tmin%d_T2min%d.png'%(nx,ny,T_min,T_min2),dpi=300)


#if option=='6':
#select_area = input('Use entire area? (Y/N):\n')
#if select_area in ['Y','y']:
out=Parallel(n_jobs=np.min([24,njobs]))(delayed(get_N_events_h)(*[data, nx, ny, min_TRMM_precip, dTmin2h, i, area.mask, area.lon_centers, area.lat_centers]) for i in range(24))
#else:
#    nlat=np.float(input('enter northern latitude:\n'))
#    slat=np.float(input('enter southern latitude:\n'))
#    wlon=np.float(input('enter western longitude:\n'))
#    elon=np.float(input('enter eastern longitude:\n'))
#    ind_mask=np.where((area.lon_centers>elon)+(area.lon_centers<wlon)+(area.lat_centers>nlat)+(area.lat_centers<slat))
#    area.mask[ind_mask]=0
#    out=Parallel(n_jobs=np.min([24,njobs]))(delayed(get_N_events_h)(*[data,nx,ny,min_TRMM_precip,dTmin2h,consider_TRMM,i, area.mask,select_area,slat,nlat,wlon,elon]) for i in range(24))

histogram=np.zeros([24])
N_events_h=[]

for i in range(24):
    histogram[i]=out[i][0]
    N_events_h.append(out[i][1])
N_events_h=np.asarray(N_events_h)

if np.any(area.mask==0):
    outside=np.where(area.mask==0)
    for i in range(24):
        N_events_h[i][outside]=np.nan
    
#print('total number of events in this region: %d'%(np.sum(N_events3[:,:]*area.mask)))

#****************************************************************
# plot hourly distribution of events, and make a gif animation:
#****************************************************************
VM=19
for i in range(24):
    fig = plt.figure(figsize=(4.1,6))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    cs = plot_image_cartopy(area,ax,N_events_hh[i,:,:]*10/(time_factor_hh[i]*(7./24.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='%02d:00-%02d:00'%(i,i+1),lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) 
    plt.tight_layout()
    plt.savefig(folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_hh%02d.png'%(nx,ny,T_minmin,T_min,i),dpi=300)
    plt.close()
os.system('convert -delay 35 '+folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_hh*.png '%(nx,ny,T_minmin,T_min)+folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_hh.gif'%(nx,ny,T_minmin,T_min))

#if select_area in ['Y','y']:
#    ## PLOT FREQUENCY OF EVENTS CLASSIFIED BY TIME OF DAY (IN 4 INTERVALS)
#    fig=plt.figure(figsize=(14,6))
#    gs = gridspec.GridSpec(1, 4, left=0.05, right=0.99, hspace=0.2, wspace=0.05, top=0.98, bottom=0.05)
#    ax = subplot(gs[0])
#    cs = plot_image_cartopy(ax,np.sum(N_events_hh[6:12,:,:],axis=0)*10/(np.mean(time_factor_hh[6:12])*(7*6./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='06:00$-$12:00') 
#    ax = subplot(gs[1])
#    cs = plot_image_cartopy(ax,np.sum(N_events_hh[12:18,:,:],axis=0)*10/(np.mean(time_factor_hh[12:18])*(7*6./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='12:00$-$18:00',labelslat=False) 
#    ax = subplot(gs[2])
#    cs = plot_image_cartopy(ax,np.sum(N_events_hh[18:21,:,:],axis=0)*10/(np.mean(time_factor_hh[18:21])*(7*3./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='18:00$-$21:00',labelslat=False) 
#    ax = subplot(gs[3])
#    cs = plot_image_cartopy(ax,(np.sum(N_events_hh[21:,:,:],axis=0)+np.sum(N_events_hh[:6,:,:],axis=0))*10/((np.mean(time_factor_hh[21:])+np.mean(time_factor_hh[:6]))*0.5*(7*9./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='21:00$-$06:00',labelslat=False) 
#    plt.savefig('N_events_total_nxny%d%d_Tmin%d_T2min%d_time_day.png'%(nx,ny,T_min,T_min2),dpi=300)

#    if consider_TRMM:
#        VM=1.5
#    else:
#        VM=1.5
#    fig=plt.figure(figsize=(14,6))
#    gs = gridspec.GridSpec(1, 4, left=0.05, right=0.99, hspace=0.2, wspace=0.05, top=0.98, bottom=0.05)
#    ax = subplot(gs[0])
#    N_events_h_smooth = gaussian_filter(np.sum(N_events_h[6:12,:,:],axis=0), sigma=1, mode='reflect', cval=0.0 )
#    N_events_h_smooth[-2,:] = np.nan
#    N_events_h_smooth[:,1] = np.nan
#    cs=plot_image_cartopy(ax,N_events_h_smooth*1e4/((ndays*6./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,0.2), cmap='afmhot_r',label='x10$^{-4}$ day$^{-1}$km$^{-2}$',remove_borders=True, title='06:00$-$12:00')
#    ax = subplot(gs[1])
#    N_events_h_smooth = gaussian_filter(np.sum(N_events_h[12:18,:,:],axis=0), sigma=1, mode='reflect', cval=0.0 )
#    N_events_h_smooth[-2,:] = np.nan
#    N_events_h_smooth[:,1] = np.nan
#    cs=plot_image_cartopy(ax,N_events_h_smooth*1e4/((ndays*6./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,0.2), cmap='afmhot_r',label='x10$^{-4}$ day$^{-1}$km$^{-2}$',remove_borders=True, title='12:00$-$18:00',labelslat=False)
#    ax = subplot(gs[2])
#    N_events_h_smooth = gaussian_filter(np.sum(N_events_h[18:21,:,:],axis=0), sigma=1, mode='reflect', cval=0.0 )
#    N_events_h_smooth[-2,:] = np.nan
#    N_events_h_smooth[:,1] = np.nan
#    cs=plot_image_cartopy(ax,N_events_h_smooth*1e4/((ndays*3./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,0.2), cmap='afmhot_r',label='x10$^{-4}$ day$^{-1}$km$^{-2}$',remove_borders=True, title='18:00$-$21:00',labelslat=False)
#    ax = subplot(gs[3])
#    N_events_h_smooth = gaussian_filter(np.sum(N_events_h[-3:,:,:],axis=0)+np.sum(N_events_h[:6,:,:],axis=0), sigma=1, mode='reflect', cval=0.0 )
#    N_events_h_smooth[-2,:] = np.nan
#    N_events_h_smooth[:,1] = np.nan
#    cs=plot_image_cartopy(ax,N_events_h_smooth*1e4/((ndays*9./24.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,0.2), cmap='afmhot_r',label='x10$^{-4}$ day$^{-1}$km$^{-2}$',remove_borders=True, title='21:00$-$06:00',labelslat=False) 
#    plt.savefig('N_events_nxny%d%d_Tmin%d_T2min%d_time_day.png'%(nx,ny,T_min,T_min2),dpi=300)

#*************************************************
# plot hourly distribution (in 2 hour intervals):
VM=28.
fig=plt.figure(figsize=(13.,5.9))
gs = gridspec.GridSpec(2, 7, left=0.03, right=0.95, hspace=0.2, wspace=0.005, top=0.95, bottom=0.05,width_ratios=[1,1,1,1,1,1,0.1])
ind_plots=[0,1,2,3,4,5,7,8,9,10,11,12]
hours=['0000-0200','0200-0400','0400-0600','0600-0800','0800-1000','1000-1200','1200-1400','1400-1600','1600-1800','1800-2000','2000-2200','2200-2400']

N_events_h2 = (N_events_hh[1:,:,:]+N_events_hh[:-1,:,:])[::2,:,:]
time_factor_hh2 = (0.5*(time_factor_hh[1:]+time_factor_hh[:-1]))[::2]
for i in range(12):
    ax = subplot(gs[ind_plots[i]],projection=ccrs.PlateCarree())
    if ind_plots[i] in [0,7]:
        labelslat=True
    else:
        labelslat=False
    if ind_plots[i]>6:
        labelslon=True
    else:
        labelslon=False
    N_events_h_smooth = N_events_h2[i,:,:]
    cs = plot_image_cartopy(area,ax,N_events_h_smooth*10/(np.mean(time_factor_hh2[i])*(7./12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM+0.01,2), cmap='afmhot_r',cb=False,label='',remove_borders=True, title=hours[i],labelslon=labelslon,labelslat=labelslat,fslonlat=10,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
cax = subplot(gs[0:2,6])
cax.set_position([0.94,0.2,0.01,0.6])
cbar = plt.colorbar(cs, pad=-2, extend='max', ticks=np.arange(0,VM+0.01,5), orientation='vertical',cax=cax)
cbar.set_label(label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',size=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig(folder+'/Fig6.png',dpi=300)

#***********************************
# Now get the montly distribution:
#***********************************
#if select_area in ['Y','y']:
out = Parallel(n_jobs=np.min([12,njobs]))(delayed(get_N_events_m)(*[data,nx,ny,min_TRMM_precip,dTmin2h,i,area.mask,area.lon_centers,area.lat_centers]) for i in range(12))
#else:    
#    out = Parallel(n_jobs=np.min([12,njobs]))(delayed(get_N_events_m)(*[data,nx,ny,min_TRMM_precip,dTmin2h,consider_TRMM,i,area.mask,select_area,slat,nlat,wlon,elon]) for i in range(12))

histogram_m=np.zeros([12])
N_events_m=[]
for i in range(12):
    histogram_m[i]=out[i][0]
    N_events_m.append(out[i][1])
N_events_m = np.asarray(N_events_m)
if np.any(area.mask==0):
    outside=np.where(area.mask==0)
    for i in range(12):
        N_events_m[i][outside]=np.nan

ft = abs(np.fft.rfft(moving_avg(histogram_m)))
print('S0=log(f0/(f1+f2))=%.4f'%(np.log(ft[0]/(ft[1]+ft[2]))))
print('S1=log(f1/f2)=%.4f'%(np.log(ft[1]/ft[2])))

#************************************************
# plot each month, and make a gif animation:
#************************************************
VM=13
months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
for i in range(12):
    fig = plt.figure(figsize=(4.1,6))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    cs = plot_image_cartopy(area,ax,N_events_mm[i,:,:]*10/((7/12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title=months[i],lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
    plt.tight_layout()
    plt.savefig(folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_mm%02d.png'%(nx,ny,T_minmin,T_min,i+1),dpi=300)
    plt.close()
os.system('convert -delay 40 '+folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_mm*.png '%(nx,ny,T_minmin,T_min)+folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_mm.gif'%(nx,ny,T_minmin,T_min))
os.system('rm '+folder+'/N_events_total_nxny%d%d_Tmin%d_T2min%d_mm*.png'%(nx,ny,T_minmin,T_min))

## PLOT FREQUENCY OF EVENTS BY MONTH:
fig=plt.figure(figsize=(13.,5.9))
gs = gridspec.GridSpec(2, 7, left=0.03, right=0.95, hspace=0.2, wspace=0.005, top=0.95, bottom=0.05,width_ratios=[1,1,1,1,1,1,0.1])
#gs = gridspec.GridSpec(1, 4, left=0.04, right=0.99, hspace=0.2, wspace=0.05, top=0.99, bottom=0.05)
ind_plots=[0,1,2,3,4,5,7,8,9,10,11,12]
month=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
def plot_months():
    for i in range(12):
        ax = subplot(gs[ind_plots[i]],projection=ccrs.PlateCarree())
        if ind_plots[i] in [0,7]:
            labelslat=True
        else:
            labelslat=False
        if ind_plots[i]>6:
            labelslon=True
        else:
            labelslon=False
        cs = plot_image_cartopy(area,ax,N_events_mm[i,:,:]*10/((7/12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',cb=False,label='',remove_borders=True, title=month[i],labelslon=labelslon,labelslat=labelslat,fslonlat=10,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
        #plt.annotate('%d'%(i+1),xy=(0.1,0.8),xycoors='axes fraction',fontsize=14)
plot_months()
cax = subplot(gs[0:2,6])
cax.set_position([0.94,0.2,0.01,0.6])
cbar = plt.colorbar(cs, pad=-2, extend='max', ticks=np.arange(0,VM,2), orientation='vertical',cax=cax)
cbar.set_label(label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',size=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig(folder+'/Fig3.png',dpi=300)
#    plt.savefig('N_events_total_nxny%d%d_Tmin%d_T2min%d_months.png'%(nx,ny,T_min,T_min2),dpi=300)

#    fig=plt.figure(figsize=(14,5.9))
#    gs = gridspec.GridSpec(1, 4, left=0.035, right=0.99, hspace=0.2, wspace=0.05, top=0.99, bottom=0.13)
#    ax = subplot(gs[0],projection=ccrs.PlateCarree())
#    cs = plot_image_cartopy(ax,(N_events_mm[-1,:,:]+np.sum(N_events_mm[:2,:,:],axis=0))*10/((7/4.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='DJF')
#    ax = subplot(gs[1],projection=ccrs.PlateCarree())
#    cs = plot_image_cartopy(ax,np.sum(N_events_mm[2:5,:,:],axis=0)*10/((7/4.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='MAM', labelslat=False)
#    ax = subplot(gs[2],projection=ccrs.PlateCarree())
#    cs = plot_image_cartopy(ax,np.sum(N_events_mm[5:8,:,:],axis=0)*10/((7/4.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='JJA', labelslat=False)
#    ax = subplot(gs[3],projection=ccrs.PlateCarree())
#    cs = plot_image_cartopy(ax,np.sum(N_events_mm[8:11,:,:],axis=0)*10/((7/4.)*dx*dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='SON', labelslat=False)
#    plt.savefig('N_events_total_nxny%d%d_Tmin%d_T2min%d_seasonal.png'%(nx,ny,T_min,T_min2),dpi=300)


if np.max(histogram<100):
    scale=1
elif np.max(histogram>=1e3):
    scale=1e-3
else:
    scale=1e-2
fig=plt.figure(figsize=(5,5.5))
gs = gridspec.GridSpec(2, 1, left=0.11, right=0.99, hspace=0.3, wspace=0.2, top=0.97, bottom=0.1)
ax = subplot(gs[0])
plt.bar(np.arange(1,12.5),histogram_m*scale)
plt.xlabel('month',fontsize=16,labelpad=1)
plt.xlim(0,13)
if scale==1e-3:
    fig.text(0.005, 0.5, 'Number of events (X 10$^3$)', va='center', rotation='vertical',fontsize=16)
    #plt.ylabel('Number of events (X 10$^3$)',fontsize=16)
elif scale==1e-2:
    fig.text(0.02,0.5,'Number of events (X 10$^2$)', va='center', rotation='vertical',fontsize=16)
elif scale==1:
    fig.text(0.015,0.5,'Number of events', va='center', rotation='vertical',fontsize=16)
    #plt.ylabel('Number of events',fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1,13),fontsize=14)
plt.annotate('a)',xy=(0.15,0.9),xycoords='figure fraction',fontsize=18)

ax=subplot(gs[1])
plt.bar(np.arange(0.5,24.5),histogram*scale/time_factor_hh)
plt.xlabel('local time (h)',fontsize=16)
plt.xlim(0,24)
plt.xticks(np.arange(0,24,3),fontsize=14)
plt.yticks(np.arange(0,9,2),fontsize=14)
plt.annotate('b)',xy=(0.15,0.41),xycoords='figure fraction',fontsize=18)
#if select_area in ['Y','y']:
plt.savefig(folder+'/Fig2.png',dpi=300)
#    plt.savefig('histograms_nxny%d%d_Tmin%d_T2min%d.png'%(nx,ny,T_min,T_min2),dpi=300)
#else:
#    plt.savefig('histograms_nxny%d%d_%d.%d.%d.%d_Tmin%d_T2min%d.png'%(nx,ny,np.int(nlat),np.int(slat),np.int(wlon),np.int(elon),T_min,T_min2),dpi=300)

#plt.show()

#if option=='7':
#selected areas:
# get an estimate of seasonal regime (monomodal or bimodal) based on a fourier transform and the log of the ratio between the two amplitudes of the relevant frequencies
#data=np.load('data_nxny%d%d_Tmin%d_T2min%d.npy'%(nx,ny,T_min,T_min2))
box_size=7 # size (in gridboxes) of each size of the subdomain to use for the monthly histogram of events of each gridpoint
select='N'
S0 = np.ones([nx,ny])*np.nan
S1 = np.ones([nx,ny])*np.nan
S2 = np.ones([nx,ny])*np.nan
month_max_1 = np.ones([nx,ny])*np.nan
month_max_2 = np.ones([nx,ny])*np.nan
for j in range(area.ny-box_size+1):
    print(j,end='\r')
    #lon_w = area.lon_corners[:,0][i]
    #lon_e = area.lon_corners[:,0][i+box_size]
    lat_s = area.lat_corners[0,:][j]
    lat_n = area.lat_corners[0,:][j+box_size]
    out = Parallel(n_jobs=(area.nx-box_size+1))(delayed(get_histograms_box)(*[data,lat_s,lat_n,area.lon_corners[:,0][i],area.lon_corners[:,0][i+box_size],min_TRMM_precip,dTmin2h]) for i in range(area.nx-box_size+1))
    for i in range(area.nx-box_size+1):
        ft = abs(np.fft.rfft(moving_avg(out[i][0])))
        S0[i+np.int(0.5*(box_size-1)),j+np.int(0.5*(box_size-1))]=np.log(ft[0]/(ft[1]+ft[2]))
        S1[i+np.int(0.5*(box_size-1)),j+np.int(0.5*(box_size-1))]=np.log(ft[1]/ft[2])#(ft[1]-ft[2])/ft[0]
        S2[i+np.int(0.5*(box_size-1)),j+np.int(0.5*(box_size-1))]=np.log(ft[2]/ft[1])
        if (ft[1]/ft[2])>1.5:
            month_max_1[i+np.int(0.5*(box_size-1)),j+np.int(0.5*(box_size-1))]=out[i][1]
        if (ft[2]/ft[1])>1.5:
            month_max_2[i+np.int(0.5*(box_size-1)),j+np.int(0.5*(box_size-1))]=out[i][1]
print('')
fig=plt.figure(figsize=(13.3,5.8))
gs=gridspec.GridSpec(12, 6, left=0.03, right=0.99, hspace=0.11, wspace=0.01, top=0.95, bottom=0.095, width_ratios=[1.5,1.5,0.25,1.05,0.14,1.05])#,height_ratios=[h0,h1,h0,h1,h0,h1,h0,h1,h0,h1,h0,h1])
gs0=[]
gs0.append(gs[0:2,3])
gs0.append(gs[0:2,5])
gs0.append(gs[2:4,3])
gs0.append(gs[2:4,5])
gs0.append(gs[4:6,3])
gs0.append(gs[4:6,5])
gs0.append(gs[6:8,3])
gs0.append(gs[6:8,5])
gs0.append(gs[8:10,3])
gs0.append(gs[8:10,5])
gs0.append(gs[10:12,3])
gs0.append(gs[10:12,5])
ylims=[(0,2.95),(0,5),(0,8), (0,5.5), (0,5), (0,13), (0,5), (0,17), (0,21), (0,12), (0,7),(0,7)]
yticks=[[1,2], [2,4], [3,6],  [2,4], [2,4],  [5,10], [2,4], [6,12], [8,16],[5,10], [3,6],  [3,6]]
for i in range(len(boxes)):
    scale=1e-1
    histogram_box, month_max = get_histograms_box(data,boxes[i][2],boxes[i][3],boxes[i][0],boxes[i][1],min_TRMM_precip,dTmin2h)
    ax=fig.add_subplot(gs0[i])
    plt.bar(np.arange(1,12.5),histogram_box*scale,width=0.6)
    plt.xticks(np.arange(1,13),fontsize=10)
    plt.yticks(ticks=yticks[i],fontsize=10)
    plt.xlim(0.5,12.5)
    plt.ylim(ylims[i])
    plt.annotate('%d'%(i+1),xy=(0.075,0.75),xycoords='axes fraction',fontsize=14,fontstyle='italic')
    if i<10:
        ax.xaxis.set_major_formatter(NullFormatter())
    else:
        plt.xlabel('month',fontsize=12,labelpad=1)
fig.text(0.56, 0.5, 'Number of events ($\\times10$)', va='center', rotation='vertical',fontsize=16)
fig.text(0.78, 0.97, 'c)', va='center', rotation='horizontal',fontsize=16)

ax = fig.add_subplot(gs[0:11,0],projection=ccrs.PlateCarree())
cs=plot_image_cartopy(area,ax,S0,vmin=0,vmax=2, ticks=np.arange(-5,6,1), cmap='binary',label='$S0$',remove_borders=False, extend='both',title='a)',boxes=boxes,boxescolor=['w'],lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
ax=subplot(gs[0:11,1],projection=ccrs.Mercator(central_longitude=-75))
cs=plot_image_cartopy(area,ax,S1,vmin=-2,vmax=2, ticks=np.arange(-5,6,1), cmap='coolwarm',label='$S1$',remove_borders=False, extend='both',title='b)',labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
low_S1=np.where(S1<0)
ax.scatter(area.lon_centers[low_S1],area.lat_centers[low_S1],transform=ccrs.PlateCarree(),marker='o',s=1,c='k',edgecolor='none')
ax.annotate('...................................',xy=(0.08,-0.07),xycoords='axes fraction',zorder=10,fontsize=9)
ax.annotate('...................................',xy=(0.08,-0.08),xycoords='axes fraction',zorder=10,fontsize=9)
plt.savefig(folder+'/Fig4.png',dpi=300)
#plt.savefig('seasonality_nxny%d%d_Tmin%d_T2min%d.png'%(nx,ny,T_min,T_min2),dpi=300)
plt.show()


