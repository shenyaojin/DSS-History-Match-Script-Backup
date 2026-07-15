import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Well:

    def __init__(self):
        self.gauge_length = None
        self.strain_calculation_method = 'projection'
        pass

    def set_well_path(self, x,y,z):
        x,y,z = _check_well_path_input(x,y,z)
        self.x = x
        self.y = y
        self.z = z
        self.cal_MD()
        return self
    
    def cal_MD(self):
        x = self.x; y = self.y; z = self.z
        dists = (x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2
        dists = np.sqrt(dists)
        mds = np.cumsum(dists)
        mds = np.append(0, mds)
        self.mds = mds
    
    def calculate_displacement_along_well(self):
        # first order
        # dx = np.diff(self.x)
        # dx = np.append(dx, dx[-1])
        # dy = np.diff(self.y)
        # dy = np.append(dy, dy[-1])
        # dz = np.diff(self.z)
        # dz = np.append(dz, dz[-1])
        # second order estimation

        dx = (self.x[2:] - self.x[:-2]) / 2
        dy = (self.y[2:] - self.y[:-2]) / 2
        dz = (self.z[2:] - self.z[:-2]) / 2
        dx = dx/np.sqrt(dx**2 + dy**2 + dz**2)
        dy = dy/np.sqrt(dx**2 + dy**2 + dz**2)
        dz = dz/np.sqrt(dx**2 + dy**2 + dz**2)
        self.strain = ((self.u1[2:] - self.u1[:-2])*dx 
                       + (self.u2[2:] - self.u2[:-2])*dy 
                       + (self.u3[2:] - self.u3[:-2])*dz)/(self.mds[2:] - self.mds[:-2])
        self.strain = np.concatenate(([self.strain[0]], self.strain, [self.strain[-1]]))

        dx = np.concatenate(([dx[0]], dx, [dx[-1]]))
        dy = np.concatenate(([dy[0]], dy, [dy[-1]]))
        dz = np.concatenate(([dz[0]], dz, [dz[-1]]))
        self.disp = (self.u1*dx + self.u2*dy + self.u3*dz)

    def calculate_strain(self):
        # self.calculate_displacement_along_well()
        if self.strain_calculation_method == 'projection':
            self.calculate_displacement_along_well()
            if self.gauge_length is not None:
                GL = self.gauge_length
                smstrain = np.zeros_like(self.strain)
                for i in range(len(self.strain)):
                    ind = (self.mds >= self.mds[i]-GL/2) & (self.mds <= self.mds[i]+GL/2)
                    smstrain[i] = np.mean(self.strain[ind])
                self.strain = smstrain
        elif self.strain_calculation_method == 'fiber_length':
            if self.gauge_length is None:
                # self.strain = (self.disp[2:] - self.disp[:-2])/(self.mds[2:] - self.mds[:-2])
                base_dist = np.sqrt((self.x[2:] - self.x[:-2])**2 
                                    + (self.y[2:] - self.y[:-2])**2 
                                    + (self.z[2:] - self.z[:-2])**2)
                current_dist = np.sqrt(((self.x[2:] + self.u1[2:]) - (self.x[:-2] + self.u1[:-2]))**2
                    + ((self.y[2:] + self.u2[2:]) - (self.y[:-2] + self.u2[:-2]))**2 
                        + ((self.z[2:] + self.u3[2:]) - (self.z[:-2] + self.u3[:-2]))**2)
                self.strain = (current_dist - base_dist)/base_dist
                self.strain = np.append(self.strain, self.strain[-1])
                self.strain = np.append(self.strain[0], self.strain)
            else:
                GL = self.gauge_length
                self.strain = np.zeros_like(self.mds)
                fx = interp1d(self.mds, self.x)
                fy = interp1d(self.mds, self.y)
                fz = interp1d(self.mds, self.z)
                fu1 = interp1d(self.mds, self.u1)
                fu2 = interp1d(self.mds, self.u2)
                fu3 = interp1d(self.mds, self.u3)
                ind = np.where((self.mds>GL/2) & (self.mds<self.mds[-1]-GL/2))[0]
                for i in ind:
                    mds1 = self.mds[i] - GL/2
                    mds2 = self.mds[i] + GL/2
                    x1 = fx(mds1); x2 = fx(mds2)
                    y1 = fy(mds1); y2 = fy(mds2)
                    z1 = fz(mds1); z2 = fz(mds2)
                    u1_1 = fu1(mds1); u1_2 = fu1(mds2)
                    u2_1 = fu2(mds1); u2_2 = fu2(mds2)
                    u3_1 = fu3(mds1); u3_2 = fu3(mds2)
                    base_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    current_dist = np.sqrt((x2+u1_2 - x1-u1_1)**2 + (y2+u2_2 - y1-u2_1)**2 + (z2+u3_2 - z1-u3_1)**2)
                    self.strain[i] = (current_dist - base_dist)/base_dist

    
    def draw_data(self,color='red'):
        line = go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='lines'
                            , line=dict(color=color, width=4)
                            , name='well path')
        return line
    
    def plot_strain(self, xaxis = 'mds'):
        plt.plot(getattr(self,xaxis), self.strain)

def set_well_by_points(control_points, N = 100, smooth = None):
    control_points = np.array(control_points)
    diff_coors = np.diff(control_points,axis=0)
    # xs = np.linspace(control_points[0,0], control_points[1,0], N)
    # ys = np.linspace(control_points[0,1], control_points[1,1], N)   
    # zs = np.linspace(control_points[0,2], control_points[1,2], N)
    # for i in range(1, control_points.shape[0]-1):
    #     xs = np.append(xs, np.linspace(control_points[i,0], control_points[i+1,0], N))
    #     ys = np.append(ys, np.linspace(control_points[i,1], control_points[i+1,1], N))
    #     zs = np.append(zs, np.linspace(control_points[i,2], control_points[i+1,2], N))
    xs = control_points[:,0]
    ys = control_points[:,1]
    zs = control_points[:,2]
    mds = np.cumsum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2))
    mds = np.append(0, mds)
    sparse_mds = np.linspace(0, mds[-1], N)
    xs = np.interp(sparse_mds, mds, xs)
    ys = np.interp(sparse_mds, mds, ys)
    zs = np.interp(sparse_mds, mds, zs)
    if smooth is not None:
        if np.mod(smooth,2)==0:
            raise ValueError('smooth must be an odd number')
        xs = _smooth(xs, smooth)
        ys = _smooth(ys, smooth)
        zs = _smooth(zs, smooth)
    well = Well()
    well.set_well_path(xs,ys,zs)
    return well

def _smooth(xs, smooth):
    xs_mid = np.convolve(xs, np.ones(smooth)/smooth, mode='valid')
    xs = np.concatenate([xs[:smooth//2], xs_mid, xs[-smooth//2+1:]])
    return xs

def _check_well_path_input(x,y,z):
    # check the largest length of the input
    N = 1
    if isinstance(x, (list, np.ndarray)):
        if N < len(x):
            N = len(x)
    if isinstance(y, (list, np.ndarray)):
        if N < len(y):
            N = len(y)
    if isinstance(z, (list, np.ndarray)):
        if N < len(z):
            N = len(z)
    if isinstance(x, (int, float)):
        x = np.ones(N)*x
    if isinstance(y, (int, float)):
        y = np.ones(N)*y
    if isinstance(z, (int, float)):
        z = np.ones(N)*z
    return x,y,z

class WellZ(Well):

    def cal_MD(self):
        x = self.x; y = self.y; z = self.z
        mds = z-z[0]
        self.mds = mds

    def calculate_strain(self):
        if self.gauge_length is None:
            self.strain = (self.u3[2:] - self.u3[:-2])/(self.mds[2:] - self.mds[:-2])
            self.strain = np.append(self.strain, self.strain[-1])
            self.strain = np.append(self.strain[0], self.strain)