import numpy as np
from joblib import Parallel, delayed
import datetime as dt
import pdb


class Grid(object):
    def __init__( self, Slat, Elon, Nlat, Wlon, nx, ny, ER=6378, UTC=-5 ):
        self.njobs  = 1         # Number of jobs to parallelize the gridding of the data (set to 1 now because parallelization is done elsewhere - faster)
        self.lrlon  = Elon
        self.lrlat  = Slat
        self.urlon  = Elon
        self.urlat  = Nlat
        self.ullon  = Wlon
        self.ullat  = Nlat
        self.lllon  = Wlon
        self.lllat  = Slat

        self.ER     = ER        # Earth Radius at particular region (for computing distances from lat lon coordinates)
        self.UTC_OFFSET = UTC   # Conversion from UTC to local time (-5 for Colombian local time)
        self._measure_grid()
        self.maxlat = np.max([self.ullat,self.urlat])+1
        self.minlat = np.min([self.lllat,self.lrlat])-1
        self.minlon = np.min([self.lllon,self.ullon])-1
        self.maxlon = np.max([self.lrlon,self.urlon])+1
        self.nx     = nx
        self.ny     = ny
        self._create_grid()
        self._create_regular_grid()
        self.dx= ((self.lrlon-self.lllon)*np.pi/180.)*self.ER/nx
        self.dy= ((self.urlat-self.lrlat)*np.pi/180.)*self.ER/ny
        print('gridboxes are %.2f x %.2f km\n'%(self.dx,self.dy))

    def _measure_grid(self):
        """
        Measures the size of the grid at its eastern and northern borders,
        in degrees and km.
        """
        A1y=self.lrlat
        A1x=self.lrlon
        A2y=self.urlat
        A2x=self.urlon
        A3x=self.ullon
        A3y=self.ullat
        A4x=self.lllon
        A4y=self.lllat
        Ea_r=self.ER
        self.y_length_deg=np.sqrt((A2x-A1x)**2+(A2y-A1y)**2)
        self.y_length_km=Ea_r*self.y_length_deg*np.pi/180.
        self.x_length_deg=np.sqrt((A2x-A3x)**2+(A2y-A3y)**2)
        self.x_length_km=Ea_r*self.x_length_deg*np.pi/180.
        print('domain size (at northern and eastern borders): %.1f x %.1f km'%(self.x_length_km,self.y_length_km))

    def _create_grid(self):
        """
        subdivides the area into (nx+1 X ny+1) corners.
        """
        lon_corners=np.ones([self.nx+1,self.ny+1])*np.nan
        lat_corners=np.ones([self.nx+1,self.ny+1])*np.nan
        lon_corners[0,0]    = self.lllon
        lon_corners[-1,0]   = self.lrlon
        lon_corners[0,-1]   = self.ullon
        lon_corners[-1,-1]  = self.urlon
        self.dx=self.x_length_deg/self.nx
        self.dy=self.y_length_deg/self.ny
        for j in range(self.ny-1):
            lon_corners[0,j+1]=lon_corners[0,0]+(j+1.)*(self.ullon-self.lllon)/self.ny
            lon_corners[-1,j+1]=lon_corners[-1,0]+(j+1.)*(self.ullon-self.lllon)/self.ny
        for j in range(self.ny+1):
            for i in range(self.nx-1):
                lon_corners[i+1,j]=lon_corners[0,j]+(i+1.)*(self.lrlon-self.lllon)/self.nx
        self.lon_corners=lon_corners

        lat_corners[0,0]    = self.lllat 
        lat_corners[-1,0]   = self.lrlat
        lat_corners[0,-1]   = self.ullat
        lat_corners[-1,-1]  = self.urlat
        for j in range(self.ny-1):
            lat_corners[0,j+1]=lat_corners[0,0]+(j+1.)*(self.ullat-self.lllat)/self.ny
            lat_corners[-1,j+1]=lat_corners[-1,0]+(j+1.)*(self.ullat-self.lllat)/self.ny
        for j in range(self.ny+1):
            for i in range(self.nx-1):
                lat_corners[i+1,j]=lat_corners[0,j]+(i+1.)*(self.lrlat-self.lllat)/self.nx
        self.lat_corners=lat_corners
        lat_centers=np.ones([self.nx,self.ny])*np.nan
        lon_centers=np.ones([self.nx,self.ny])*np.nan
        for j in range(self.ny):
            for i in range(self.nx):
                lat_centers[i,j]=np.mean(lat_corners[i:i+2,j:j+2])
                lon_centers[i,j]=np.mean(lon_corners[i:i+2,j:j+2])
        self.lat_centers=lat_centers
        self.lon_centers=lon_centers
        self.mask=np.ones([self.nx,self.ny]) #default mask (nothing masked!)

    def create_mask(self, mask):
        """
        in case part of the domain wants to be masked
        """
        self.mask=mask

    def is_in_index_straight_grid(self, lon, lat):
        """
        given a pair of lon lat coordinates, it returns the (i,j) index of the grid centers where this
        point is contained. If not contained in any grid box, it renturns (np.nan, np.nan)
        This works faster than is_in_index_general() but only works if the grid is oriented Norht-South East-West.
        """
        lat0_ind=np.where(self.lat_corners[0,:]<lat)[0]
        lon0_ind=np.where(self.lon_corners[:,0]<lon)[0]
        if len(lat0_ind)>0 and lat<self.lat_corners[0,-1] and len(lon0_ind)>0 and lon<self.lon_corners[-1,0]:
            lat0_ind=lat0_ind[-1]
            lon0_ind=lon0_ind[-1]
            return ((lon0_ind,lat0_ind),True)
        else:
            return ((-9999,-9999),False)

    def is_in_index_general(self, lon, lat ):
        """
        given a pair of lon lat coordinates, it returns the (i,j) index of the grid centers where this
        point is contained. If not contained in any grid box, it renturns (np.nan, np.nan)
        """
        ishere=np.zeros_like(self.lat_centers)
        P1=(self.lon_corners[0,0],      self.lat_corners[0,0])
        P2=(self.lon_corners[-1,0],    self.lat_corners[-1,0])
        P3=(self.lon_corners[0,-1],  self.lat_corners[0,-1])
        #P4=(self.lon_corners[0,-1],    self.lat_corners[0,-1])
        if is_in_box(P1,P2,P3,(lon,lat)):
            for j in range(self.ny):
                for i in range(self.nx):
                    P1=(self.lon_corners[i,j],      self.lat_corners[i,j])
                    P2=(self.lon_corners[i+1,j],    self.lat_corners[i+1,j])
                    P3=(self.lon_corners[i,j+1],  self.lat_corners[i,j+1])
                    if is_in_box(P1,P2,P3,(lon,lat)):
                        ishere[i,j]=1
            index=np.where(ishere)
            if len(index[0])!=1 or len(index[1])!=1:
                print('Warning! lon,lat point is in more than one gridbox (?!)')
                pdb.set_trace()
            return ((np.where(ishere)[0][0],np.where(ishere)[1][0]),True)
        else:
            return ((-9999,-9999),False)

    def _create_regular_grid(self,dx=0.005,dy=0.005):
        """
        create a regular grid that covers the grid area but can be used for plotting with a NxM array. 
        dx and dy are the regular grid spacing in degrees
        """
        meshx=np.arange(self.minlon,self.maxlon+dx,dx)
        meshy=np.arange(self.minlat,self.maxlat+dy,dy)
        self.meshlon, self.meshlat = np.meshgrid(meshx,meshy)


    def extract_data(var):
        lons= var.variables['lon'][:]
        lats= var.variables['lat'][:]
        img = var.variables['data'][:]
        # get image date and time:
        date=var.variables['imageDate'][:]
        return img, lons, lats, date

    def extract_and_grid_data(self,lons,lats,img,date,time,goes_version='13'):
        """
        slice the data to use only the region that contains the study area:
        """
        temp = np.where((lons>=self.minlon)*(lons<=self.maxlon))[1]
        if len(temp)!=0:
            slicelon = (temp[0],temp[-1])
            temp = np.where((lats>=self.minlat)*(lats<=self.maxlat))[0]
            if len(temp)!=0:
                slicelat = (temp[0],temp[-1])
                img_slice = np.squeeze(img)[slicelat[0]:slicelat[1]+1,slicelon[0]:slicelon[1]+1]
                lons_slice = lons[slicelat[0]:slicelat[1]+2,slicelon[0]:slicelon[1]+2]
                lats_slice = lats[slicelat[0]:slicelat[1]+2,slicelon[0]:slicelon[1]+2]
                
                #*************************************************
                # compute brightness temperature:
                T_slice = get_T(img_slice,goes_version=goes_version)
                
                #*************************************************
                # mask all data points outside of the study area:
                mask_slice = np.ones_like(T_slice)
                P1 = (self.lon_corners[0,0], self.lat_corners[0,0])
                P2 = (self.lon_corners[-1,0], self.lat_corners[-1,0])
                P3 = (self.lon_corners[0,-1], self.lat_corners[0,-1])
                for i in range(mask_slice.shape[0]):
                    for j in range(mask_slice.shape[1]):
                        if is_in_box(P1,P2,P3,(lons_slice[i,j],lats_slice[i,j])):
                            mask_slice[i,j]=0
                T_slice_masked = np.ma.array(T_slice,mask=mask_slice)
                
                #*************************************************
                # aggregate data into the (nx,ny) grid (each gridbox 
                # will have the average brightness temperature of all 
                # data points contained:
                T_grid=np.zeros_like(self.lat_centers)
                counter=np.zeros_like(T_grid)
                if self.ullat!=self.urlat or self.lllon!=self.ullon:
                    slanted=True #slanted grid
                else:
                    slanted=False #straight grid
                jobs=[]
                jj=np.int(T_slice.shape[1]/self.njobs)
                j0=0
                for i in range(self.njobs-1):
                    jobs.append( (self,lons_slice,lats_slice,0,T_slice.shape[0],j0,j0+jj,T_slice,slanted ) )
                    j0+=jj
                jobs.append( (self,lons_slice,lats_slice,0,T_slice.shape[0],j0,T_slice.shape[1],T_slice,slanted ) )
                ( out )= Parallel(n_jobs=self.njobs)(delayed(joblib_is_in_index)(*jobs[i]) for i in range(len(jobs)))
                for i in range(len(jobs)):
                    counter=counter+out[i][0]
                    T_grid=T_grid+out[i][1]
                
                if np.any(counter==0):
                    #print '**********************\n At least one gridbox has no data in it!!!\n **********************'
                    T_grid[np.where(counter==0)]=np.nan
                    valid=np.where(counter>0)
                    T_grid[valid]=T_grid[valid]/counter[valid]
                else:
                    T_grid=T_grid/counter

                yr=np.int(date/1000)
                timestamp = dt.datetime(yr,1,1)
                #doy=date-yr*1000-1
                doy=int(date-yr*1000-1)
                timestamp += dt.timedelta(days=doy)
                hr=int(time/10000)
                mn=int((time-hr*10000)/100)
                sc=int(time-int((time/100))*100)
                timestamp += dt.timedelta(hours=hr,minutes=mn,seconds=sc)
                #convert to local time:
                timestamp += dt.timedelta(hours=self.UTC_OFFSET)

                return T_grid, T_slice_masked, lons_slice, lats_slice, timestamp, True
            else:
                return None, None, None, None, None, False
        else:
            return None, None, None, None, None, False

    def plot_area(self, path, fname=None,lllat=4,urlat=5.5,lllon=-75,urlon=-73.25,topo=True, drawgrid=True, dlatlon=0.25, plot_grid=True):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature
   
        fig=plt.figure(figsize=(7.5,9))
        projection = ccrs.LambertCylindrical(central_longitude=-75)#ccrs.LambertConformal(central_latitude=5, central_longitude=-75)
        ax = plt.axes( [0.08,0.05,0.9,0.9], projection=projection )
        lrlon=urlon
        ullon=lllon
        lrlat=lllat
        extent = [ullon,lrlon,lrlat,urlat]
        ax.set_extent(extent)
        rivers = cartopy.feature.NaturalEarthFeature(
                category='physical', name='rivers_lake_centerlines',
                scale='10m', facecolor='none', edgecolor='cornflowerblue')
        ax.add_feature(rivers, linewidth=1)
        limites_int = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        facecolor='none')
        ax.add_feature(limites_int, edgecolor='k',linewidth=0.3, linestyle='-')

        if drawgrid:
            gl=ax.gridlines(draw_labels=True, ylocs=np.arange(-2,13,2),x_inline=False, y_inline=False)
            gl.right_labels=False
            gl.top_labels=False
        if topo:
            #plot background topographic map
            from netCDF4 import Dataset 
            topo = path+'GMRTv3_6_20190507topo.grd'
            var2 = Dataset(topo, mode='r')
            xtopo=np.arange(var2.variables['x_range'][0],var2.variables['x_range'][1]+0.5*var2.variables['spacing'][0],var2.variables['spacing'][0])
            ytopo=np.arange(var2.variables['y_range'][0],var2.variables['y_range'][1]+0.5*var2.variables['spacing'][1],var2.variables['spacing'][1])
            z=var2.variables['z'][:]
            xx,yy=np.meshgrid(xtopo,ytopo)
            ret = ax.projection.transform_points(ccrs.PlateCarree(), xx, yy)
            xt = ret[..., 0]
            yt = ret[..., 1]
            z=np.reshape(z,xt.shape)[::-1,:]
            terrain_cmap()
            cs=plt.contourf(xt,yt,z,cmap='terrain_map',levels=np.arange(0,5900,100))
            cbar = plt.colorbar(cs,orientation='vertical',pad=0.05,label='elev. (m)',shrink=0.6)#,ticks=[-0.12,-0.08,-0.04,0,0.04,0.08,0.12])
            cbar.ax.tick_params(labelsize=12)

        if plot_grid:
            for i in range(self.nx+1):
                plt.plot(self.lon_corners[i,:],self.lat_corners[i,:],lw=0.5,c='b',zorder=11,transform=ccrs.PlateCarree())
            for j in range(self.ny+1):
                plt.plot(self.lon_corners[:,j],self.lat_corners[:,j],lw=0.5,c='b',zorder=11,transform=ccrs.PlateCarree())
            if np.any(self.mask==0):
                for i in range(self.nx):
                    for j in range(self.ny):
                        if self.mask[i,j]==0:
                            plt.scatter(self.lon_centers[i,j],self.lat_centers[i,j],marker='x',c='k',s=10, transform=ccrs.PlateCarree())
        if fname!=None:
            plt.savefig(fname)
        else:
            plt.show()

            
def is_in_box(P1,P2,P3,P4):
    """
    returns 1 if P4 is contained in the rectangle defined by P1, P2 and P3, 0 otherwise
    """
    AMAB=(P4[0]-P1[0])*(P3[0]-P1[0])+(P4[1]-P1[1])*(P3[1]-P1[1])
    AMAD=(P4[0]-P1[0])*(P2[0]-P1[0])+(P4[1]-P1[1])*(P2[1]-P1[1])
    if (AMAB>0)*(((P3[1]-P1[1])**2+(P3[0]-P1[0])**2)>AMAB)*(AMAD>0)*(((P2[1]-P1[1])**2+(P2[0]-P1[0])**2)>AMAD):
        return True
    else:
        return False

def get_T(img, goes_version='13'):
    """
    #****************************************************************************************
    # this function assumes GOES-13 band 4 (10.7um, detector 'a' (North?)) values of constants
    # to convert GVAR counts to a brightness temperature. 
    # GVAR counts in netcdf format are 16-bit. To convert to 10-bit divide by 32
    # (https://www.ncdc.noaa.gov/sites/default/files/attachments/Satellite-Frequently-Asked-Questions.pdf)
    # The following steps are documented in:
    # http://www.ospo.noaa.gov/Operations/GOES/calibration/gvar-conversion.html#temp
    """
    error_setting=np.seterr(invalid='ignore')  # to avoid warnings when rad<0 (will set Teff to nan anyway)
    img=img/32. # 16-bit to 10-bit conversion
   
    b=15.6854
    m=5.2285
    rad = (img-b)/m

    c1=1.191066e-5 #[mW/(m2-sr-cm-4)]
    c2=1.438833 #(K/cm-1)
    n=937.23

    Teff= (c2*n)/np.log(1.+(c1*n*n*n)/rad)
    
    if goes_version=='13':
        a=-0.386043
        b=1.001298
    elif goes_version=='14':
        a=-0.2875616
        b=1.001258
    elif goes_version=='15':
        a=-0.3433922
        b=1.001259
    
    return a+b*Teff

def joblib_is_in_index(area,lons_slice,lats_slice,i0,i1,j0,j1,T_slice,slanted):
    counter=np.zeros([area.nx,area.ny])
    T_grid=np.zeros([area.nx,area.ny])
    if slanted:
        for i in np.arange(i0,i1):
            for j in np.arange(j0,j1):
                index = area.is_in_index_general(lons_slice[i,j],lats_slice[i,j])
                if index[1]:
                    counter[index[0]]+=1.
                    T_grid[index[0]]+=T_slice[i,j]
    else:
        for i in np.arange(i0,i1):#range(T_slice.shape[0]):
            for j in np.arange(j0,j1):#range(T_slice.shape[1]):
                lat0_ind=np.where(area.lat_corners[0,:]<lats_slice[i,j])[0]
                if len(lat0_ind)>0 and lats_slice[i,j]<area.lat_corners[0,-1]:
                    lon0_ind=np.where(area.lon_corners[:,0]<lons_slice[i,j])[0]
                    if len(lon0_ind)>0 and lons_slice[i,j]<area.lon_corners[-1,0] and (T_slice[i,j]!=np.nan):
                        counter[lon0_ind[-1],lat0_ind[-1]]+=1
                        T_grid[lon0_ind[-1],lat0_ind[-1]]+=T_slice[i,j]
    return counter, T_grid

def read_job(area,lons,lats,img,date,times,goes_version):
    T_grid =[]
    time = []
    for i in range(len(lons)):
        T_gridtemp, T_slice_maskedtemp, lons_slice, lats_slice, timestamp, valid = area.extract_and_grid_data(lons[i],lats[i],img[i],date[i],times[i],goes_version[i])
        if valid:
            T_grid.append(T_gridtemp)
            time.append([timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second])
    return np.asarray(T_grid), np.asarray(time)

def terrain_cmap():
    #modify matplotlib's 'terrain' colormap to make it better
    import matplotlib.pyplot as plt
    import matplotlib.colors
    colors_land = plt.cm.terrain(np.linspace(0.3, 1, 200))
    colors=colors_land
    terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map', colors)
    plt.register_cmap(cmap=terrain_map)


