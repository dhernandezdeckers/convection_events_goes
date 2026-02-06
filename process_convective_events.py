import numpy as np
import pickle
import pdb
from convective_events_func import *
from extras import *
import datetime as dt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib.gridspec as gridspec
from pylab import *
import events_obj
from scipy.ndimage import gaussian_filter
import os
import cartopy.crs as ccrs

# **********************************************************
# case parameters (should match those used in read_GOES_data.py and find_convective_events.py)
# **********************************************************
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
nx              = int(param_dict['nx'])
ny              = int(param_dict['ny'])


#case_name   = 'NWSA_new'#'magdalena_cauca_G16'#'MM_G13'#'NWSA'#GOES16_2018-2022_HR'#orinoco_amazonas'    # optional, for file names. 
#nx          = 97#40#38#16#80#160#80#66        # study area grid size
#ny          = 123#73#55#28#106#212#106#83
#
##**********************************************************
## Parameters for convective event identification:
##**********************************************************
## threshold for brightness temperature of broader convective event area (in K):
#T_min           = 235
#
## threshold for brightness temperature that must be found within the broader area at least at one gridbox (in K):
#T_minmin        = 210
#
## threshold for minimun decrease in brightness temperature in 2 hours (in K)
#dTmin2h         = -50
#
## maximum time difference in hours between two colocated systems to be considered the same event
#dt_max          = 1
#
## minimum 3-hourly precipitation value (in mm) according to TRMM to consider events
## (set to 0 if 3-hourly TRMM data is not used as criteria to identify events,
## set to >0 if yes):
#min_TRMM_precip = 0#0.1
#
## Path where TRMM 3-hourly precipitation data (netcdf format) is located
## (only needed if min_TRMM_precip set to >0):
#TRMM_data_path  = '/media/HD3/TRMM_3HR/netcdf/'
#
## number of jobs for parallelization:
#njobs           = 48   
#
## conversion from local to UTC time (hours) (only used for TRMM data):
#UTC_offset      = 5
##**********************************************************
##**********************************************************
print_areas = False

max_sizekm2 = 300000    # convective systems larger than this will be discarded for some analyses

# coordinate limits for plots:
lllat = 0#-2.15#-4.5
urlat = 12#12.7#7
lllon = -79.95#-76
urlon = -68.9#-66.8

correct_for_data_availability = False       # if certain hours (or months) have less available data than others, this can cause certain (unreal) biases in the distributions of events. This can reduce this bias (not recommended for GOES-16 data which does not have this problem as much as GOES-13). Default: False.

limited_area = False # True if only events from a subdomain are to be processed

# this only has effect if limited_area is True:
subdomain_slat = -5#4.98#6.906134
subdomain_nlat = 13#8.41#9.410578
subdomain_wlon = -80#-74.90#-75.533732
subdomain_elon = -66#-73.19#-73.868053


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
    T_grid  = np.load(folder+'/T_grid_'+case_name+'.npy')
    time    = np.load(folder+'/time_'+case_name+'.npy')
    area    = pickle.load( open(folder+'/area_'+case_name+'.p','rb'),encoding='latin1')
    print('gridboxes are %.2f x %.2f km\n'%(area.dx,area.dy))
import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('jet')
jet2 = truncate_colormap(cmap, 0.4, 1)

# READ THE DATA PRODUCED BY find_convective_events_runscript.py:
N_events_total          = np.load(folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin)) 
N_events_hh             = np.load(folder+'/N_events_hh_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events_mm             = np.load(folder+'/N_events_mm_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events_total_Tmin     = np.load(folder+'/N_events_total_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin)) 
mean_ssize_Tmin         = np.load(folder+'/mean_ssize_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin)) 
mean_sdur_Tmin          = np.load(folder+'/mean_sdur_Tmin_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events                = np.load(folder+'/N_events_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events_wTRMM          = np.load(folder+'/N_events_wTRMM_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events_wTRMM_sizelimit= np.load(folder+'/N_events_wTRMM_sizelimit_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
N_events_wTRMM_mindTpos = np.load(folder+'/N_events_wTRMM_mindTpos_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
data                    = np.load(folder+'/data_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
data                    = data[np.where((data[:,8]<=dTmin2h)*(data[:,16]>=min_TRMM_precip))] # make sure all events have minimum threshold of dTmin2h, and min_TRMM_precip

if limited_area:
    data = data[np.where((data[:,1]>=subdomain_wlon)*(data[:,1]<=subdomain_elon)*(data[:,2]>=subdomain_slat)*(data[:,2]<=subdomain_nlat))]
    write_events_file_from_data(data, area, np.arange(data.shape[0]), folder+'/events_'+case_name+'_Tmin%d_T2min%d_subdomain_%.2f_%.2f_%.2f_%.2f.txt'%(T_min,T_minmin,subdomain_wlon,subdomain_elon,subdomain_slat,subdomain_nlat))
    #setup mask:
    area.mask[np.where(area.lon_centers<subdomain_wlon)]=0
    area.mask[np.where(area.lon_centers>subdomain_elon)]=0
    area.mask[np.where(area.lat_centers<subdomain_slat)]=0
    area.mask[np.where(area.lat_centers>subdomain_nlat)]=0

mean_ssize_minBTpeak = np.load(folder+'/mean_ssize_minBTpeak_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))
mean_sdur_minBTpeak = np.load(folder+'/mean_sdur_minBTpeak_'+case_name+'_Tmin%d_T2min%d.npy'%(T_min,T_minmin))

#ndays   = (dt.date(*time[-1][:3])-dt.date(*time[0][:3])).days # total number of days of the time period
#ndays_m = [] # number of days per month in the entire time period
#for m in range(12):
#    ndays_temp=0
#    for y in np.arange(time[0][0],time[-1][0]):
#        if m<11:
#            ndays_temp+=(dt.date(y,m+2,1)-dt.date(y,m+1,1)).days
#        else:
#            ndays_temp+=31 #december
#    ndays_m.append(ndays_temp)
#ndays_m = np.asarray(ndays_m)

deltat = find_delta_t(time)

if correct_for_data_availability:
    time_factor_hh, tot_time_factor = get_factor_available_data(T_grid, time, nx, ny, delta_t=deltat)
else:
    time_factor_hh = np.ones(24)
    tot_time_factor = 1.

time_years=data[:,-8]+(data[:,-7]-1)/12+data[:,-6]/365
time_period_years = np.max(time_years)-np.min(time_years)

fig=plt.figure(figsize=(14,5.9))
gs = gridspec.GridSpec(1, 4, left=0.035, right=0.99, hspace=0.2, wspace=0.05, top=0.99, bottom=0.13)
ax = subplot(gs[0],projection=ccrs.Mercator(central_longitude=-75))
cs=plot_image_cartopy(area,ax,N_events_total_Tmin[:,:]*10/(tot_time_factor*time_period_years*area.dx*area.dy),vmin=0,vmax=10, ticks=np.arange(0,10,2), cmap='afmhot_r',label='event rate density (BT<235K)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='a)',lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
ax = subplot(gs[1],projection=ccrs.Mercator(central_longitude=-75))
N_events3_smooth = gaussian_filter(N_events_wTRMM, sigma=1, mode='reflect', cval=0.0 )
N_events3_smooth[-2,:]=np.nan
N_events3_smooth[:,1]=np.nan
N_events3_smooth[np.where(area.mask==0)] = np.nan
cs=plot_image_cartopy(area, ax, N_events3_smooth[:,:]*10/(tot_time_factor*time_period_years*area.dx*area.dy),vmin=0,vmax=0.275, ticks=np.arange(0,0.4,0.1), cmap='afmhot_r',label='event rate density (Tmin location)\n(x10$^{-1}$ km$^{-2}$yr$^{-1}$)',remove_borders=True, title='b)',labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) # remove borders to avoid unrealistic counts at borders due to large systems that cross the boundary
ax = subplot(gs[2],projection=ccrs.Mercator(central_longitude=-75))
ssize_smooth = gaussian_filter(mean_ssize_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
ssize_smooth[np.where(area.mask==0)] = np.nan
plot_image_cartopy(area, ax, ssize_smooth*1e-3, vmin=0, vmax=38, cmap='afmhot_r', label='mean event size (x10$^3$km$^2$))', remove_borders=True, ticks=[0,10,20,30],title='c)',logscale=False, labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
ax = subplot(gs[3],projection=ccrs.Mercator(central_longitude=-75))
sdur_smooth = gaussian_filter(mean_sdur_minBTpeak, sigma=2, mode='reflect', cval=0.0 )
sdur_smooth[np.where(area.mask==0)] = np.nan
plot_image_cartopy(area, ax, sdur_smooth, vmin=0, vmax=6, cmap='afmhot_r', label='mean event duration (h)', remove_borders=True, ticks=[0,1,2,3,4,5,6],title='d)',logscale=False, labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)    
plt.savefig(folder+'/Fig1.png',dpi=300)

print('Number of events regardless of precipitation is %d'%(np.nansum(N_events)))
print('Number of events is %d'%(np.nansum(N_events_wTRMM)))
print('Number of events with size smaller than %f is %d (used for storm size and duration)'%(max_sizekm2, np.nansum(N_events_wTRMM_sizelimit)))


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

LIS_file=Dataset('/media/HD3/TRMM_LIS/lis_vhrfc_1998_2013_v01.nc',mode='r')
LIS=LIS_file.variables['VHRFC_LIS_FRD'][:]
lons=np.arange(-180,180.05,0.1)
lats=np.arange(-38,38.05,0.1)
loncorners,latcorners=np.meshgrid(lons,lats)
ax = fig.add_subplot(gs[0:11,1],projection=ccrs.PlateCarree())
plot_image_cartopy(area, ax, LIS, vmin=0, vmax=150, cmap='afmhot_r', label='FRD (fl km$^{-2}$yr$^{-1}$)', remove_borders=True, title='b)', ticks=np.arange(0,151,25),loncorners=[loncorners],latcorners=[latcorners],labelslat=False,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)

plt.savefig(folder+'/Fig5.png',dpi=300)


out=Parallel(n_jobs=np.min([24,njobs]))(delayed(get_N_events_h)(*[data, nx, ny, min_TRMM_precip, dTmin2h, i, area.mask, area.lon_centers, area.lat_centers]) for i in range(24))

histogram=np.zeros([24])
N_events_h=[]

for i in range(24):
    histogram[i]=out[i][0]
    N_events_h.append(out[i][1])
N_events_h=np.asarray(N_events_h) # this counts each event only at its minimum temperature gridpoint, not its entire area below T_min.

if np.any(area.mask==0):
    outside=np.where(area.mask==0)
    for i in range(24):
        N_events_h[i][outside]=np.nan

#****************************************************************
# plot hourly distribution of events, and make a gif animation:
#****************************************************************
VM=19
for i in range(24):
    fig = plt.figure(figsize=(4.1,6))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    cs = plot_image_cartopy(area,ax,N_events_hh[i,:,:]*10/(time_factor_hh[i]*(time_period_years/24.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,4), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title='%02d:00-%02d:00'%(i,i+1),lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True) 
    plt.tight_layout()
    plt.savefig(folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_hh%02d.png'%(T_min,T_minmin,i),dpi=300)
    plt.close()
os.system('convert -delay 35 '+folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_hh*.png '%(T_min,T_minmin)+folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_hh.gif'%(T_min,T_minmin))


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
    N_events_h_smooth[np.where(area.mask==0)]=np.nan
    cs = plot_image_cartopy(area,ax,N_events_h_smooth*10/(np.mean(time_factor_hh2[i])*(time_period_years/12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM+0.01,2), cmap='afmhot_r',cb=False,label='',remove_borders=True, title=hours[i],labelslon=labelslon,labelslat=labelslat,fslonlat=10,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
cax = subplot(gs[0:2,6])
cax.set_position([0.94,0.2,0.01,0.6])
cbar = plt.colorbar(cs, pad=-2, extend='max', ticks=np.arange(0,VM+0.01,5), orientation='vertical',cax=cax)
cbar.set_label(label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',size=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig(folder+'/Fig6.png',dpi=300)

#***********************************
# Now get the montly distribution:
#***********************************
out = Parallel(n_jobs=np.min([12,njobs]))(delayed(get_N_events_m)(*[data,nx,ny,min_TRMM_precip,dTmin2h,i,area.mask,area.lon_centers,area.lat_centers]) for i in range(12))

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
    mm = N_events_mm[i,:,:]
    mm[np.where(area.mask==0)] = np.nan
    cs = plot_image_cartopy(area,ax,mm[:,:]*10/((time_period_years/12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',remove_borders=True, title=months[i],lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
    plt.tight_layout()
    plt.savefig(folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_mm%02d.png'%(T_min,T_minmin,i+1),dpi=300)
    plt.close()
os.system('convert -delay 40 '+folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_mm*.png '%(T_min,T_minmin)+folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_mm.gif'%(T_min,T_minmin))
os.system('rm '+folder+'/N_events_total_'+case_name+'_Tmin%d_T2min%d_mm*.png'%(T_min,T_minmin))

## PLOT FREQUENCY OF EVENTS BY MONTH:
fig=plt.figure(figsize=(13.,5.9))
gs = gridspec.GridSpec(2, 7, left=0.03, right=0.95, hspace=0.2, wspace=0.005, top=0.95, bottom=0.05,width_ratios=[1,1,1,1,1,1,0.1])
ind_plots=[0,1,2,3,4,5,7,8,9,10,11,12]
month=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
def plot_months():
    number_years = time[-1,0]-time[0,0]+1 # for averages, this is the number of years considered
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
        mm = N_events_mm[i,:,:]
        mm[np.where(area.mask==0)]=np.nan
        cs = plot_image_cartopy(area,ax,mm[:,:]*10/((number_years/12.)*area.dx*area.dy),vmin=0,vmax=VM, ticks=np.arange(0,VM,2), cmap='afmhot_r',cb=False,label='',remove_borders=True, title=month[i],labelslon=labelslon,labelslat=labelslat,fslonlat=10,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon,topo=True)
plot_months()
cax = subplot(gs[0:2,6])
cax.set_position([0.94,0.2,0.01,0.6])
cbar = plt.colorbar(cs, pad=-2, extend='max', ticks=np.arange(0,VM,2), orientation='vertical',cax=cax)
cbar.set_label(label='x10$^{-1}$ km$^{-2}$yr$^{-1}$',size=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig(folder+'/Fig3.png',dpi=300)

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
elif scale==1e-2:
    fig.text(0.02,0.5,'Number of events (X 10$^2$)', va='center', rotation='vertical',fontsize=16)
elif scale==1:
    fig.text(0.015,0.5,'Number of events', va='center', rotation='vertical',fontsize=16)
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
plt.savefig(folder+'/Fig2.png',dpi=300)

box_size=7 # size (in gridboxes) of each size of the subdomain to use for the monthly histogram of events of each gridpoint
select='N'
S0 = np.ones([nx,ny])*np.nan
S1 = np.ones([nx,ny])*np.nan
S2 = np.ones([nx,ny])*np.nan
month_max_1 = np.ones([nx,ny])*np.nan
month_max_2 = np.ones([nx,ny])*np.nan
for j in range(area.ny-box_size+1):
    print(j,end='\r')
    lat_s = area.lat_corners[0,:][j]
    lat_n = area.lat_corners[0,:][j+box_size]
    out = Parallel(n_jobs=(area.nx-box_size+1))(delayed(get_histograms_box)(*[data,lat_s,lat_n,area.lon_corners[:,0][i],area.lon_corners[:,0][i+box_size],min_TRMM_precip,dTmin2h]) for i in range(area.nx-box_size+1))
    for i in range(area.nx-box_size+1):
        ft = abs(np.fft.rfft(moving_avg(out[i][0])))
        S0[i+int(0.5*(box_size-1)),j+int(0.5*(box_size-1))]=np.log(ft[0]/(ft[1]+ft[2]))
        S1[i+int(0.5*(box_size-1)),j+int(0.5*(box_size-1))]=np.log(ft[1]/ft[2])#(ft[1]-ft[2])/ft[0]
        S2[i+int(0.5*(box_size-1)),j+int(0.5*(box_size-1))]=np.log(ft[2]/ft[1])
        if (ft[1]/ft[2])>1.5:
            month_max_1[i+int(0.5*(box_size-1)),j+int(0.5*(box_size-1))]=out[i][1]
        if (ft[2]/ft[1])>1.5:
            month_max_2[i+int(0.5*(box_size-1)),j+int(0.5*(box_size-1))]=out[i][1]
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
plt.show()

