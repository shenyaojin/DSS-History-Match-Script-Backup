import numpy as np
from copy import deepcopy
from . import Element
import plotly.graph_objects as go


class FractureZ:
    """
    This class calculate the displacement in the local coordinate system of the fracture.
    Monitor well coordinates must be in the local coordinate system of the fracture.
    """

    def __init__(self):
        self.elements = None
        self.monitor_wells = None
    
    def set_monitor_wells(self, monitor_wells):
        if not isinstance(monitor_wells, list):
            monitor_wells = [monitor_wells]
        self.monitor_wells = deepcopy(monitor_wells)
        return self

    def set_fracture_elements(self, elements):
        if not isinstance(elements, list):
            elements = [elements]
        self.elements = elements
        return self

    def u1(self, x1, x2, x3):
        u1 = 0
        for element in self.elements:
            relative_x1 = x1 - element.abs_x1
            relative_x2 = x2 - element.abs_x2
            relative_x3 = x3 - element.abs_x3
            element.set_coors(relative_x1, relative_x2, relative_x3)
            u1 += element.u1
        return u1

    def u2(self, x1, x2, x3):
        u2 = 0
        for element in self.elements:
            relative_x1 = x1 - element.abs_x1
            relative_x2 = x2 - element.abs_x2
            relative_x3 = x3 - element.abs_x3
            element.set_coors(relative_x1, relative_x2, relative_x3)
            u2 += element.u2
        return u2

    def u3(self, x1, x2, x3):
        u3 = 0
        for element in self.elements:
            relative_x1 = x1 - element.abs_x1
            relative_x2 = x2 - element.abs_x2
            relative_x3 = x3 - element.abs_x3
            element.set_coors(relative_x1, relative_x2, relative_x3)
            u3 += element.u3
        return u3
    
    def calculate_well_displacement(self):
        for well in self.monitor_wells:
            well.u1 = self.u1(well.x, well.y, well.z)
            well.u2 = self.u2(well.x, well.y, well.z)
            well.u3 = self.u3(well.x, well.y, well.z)
            well.calculate_strain()
    
    def calculate(self):
        self.calculate_well_displacement()
    
    def draw_data(self):
        data = []
        # draw monitor wells
        for well in self.monitor_wells:
            data.append(well.draw_data())
        
        # draw fracture elements
        for element in self.elements:
            x1,x2,x3 = element.get_plot_coors()
            mesh = go.Mesh3d(x=x1, y=x2, z=x3, color='blue', opacity=0.5)
            data.append(mesh)
        return data
    
    def draw(self):
        fig = go.Figure(data=self.draw_data())
        fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        eye=dict(x=0, y=-1.5, z=0.5)
                    )
                )
            )
        return fig
    
    def plot_strain(self,**kwargs):
        for well in self.monitor_wells:
            well.plot_strain(**kwargs)

def get_ellipse_fracture(length, height_ratio, width_ratio, S1_ratio, S2_ratio, element_N):
    '''
    This function returns an elliptical fracture
    get_ellipse_fracture(length, height_ratio, width_ratio, S1_ratio, S2_ratio, element_N):
    return -> FractureZ instance
    '''
    fracture_length = length
    elem_length = fracture_length / element_N
    elem_x = np.arange(element_N) * elem_length - fracture_length / 2 + elem_length / 2
    elems = []
    for i in range(element_N):
        elem_height = (fracture_length**2/4-(elem_x[i])**2)**0.5 * height_ratio
        elem_width = (fracture_length**2/4-(elem_x[i])**2)**0.5 * width_ratio
        S1 = (fracture_length**2/4-(elem_x[i])**2)**0.5 * S1_ratio
        S2 = (fracture_length**2/4-(elem_x[i])**2)**0.5 * S2_ratio
        elem = Element.Element(elem_length, elem_height, elem_width, S1, S2)
        elem.set_global_coors(elem_x[i], 0, 0)
        elems.append(elem)
    frac = FractureZ().set_fracture_elements(elems)
    return frac

def get_ellipse_asymmetric_updown(half_length, up_ratio, down_ratio, width_ratio, S1_ratio, S2_ratio, element_N):
    fracture_length = half_length*2
    elem_length = fracture_length / element_N
    elem_x = np.arange(element_N) * elem_length - fracture_length / 2 + elem_length / 2
    elems = []
    for i in range(element_N):
        up = (fracture_length**2/4-(elem_x[i])**2)**0.5 * up_ratio
        down = (fracture_length**2/4-(elem_x[i])**2)**0.5 * down_ratio
        elem_width = (fracture_length**2/4-(elem_x[i])**2)**0.5 * width_ratio
        S1 = (fracture_length**2/4-(elem_x[i])**2)**0.5 * S1_ratio
        S2 = (fracture_length**2/4-(elem_x[i])**2)**0.5 * S2_ratio
        elem = Element.Element(elem_length, up+down, elem_width, S1, S2)
        elem.set_global_coors(elem_x[i], (up-down)/2, 0)
        elems.append(elem)
    frac = FractureZ().set_fracture_elements(elems)
    return frac


def fractureZ_to_global(frac, strike, dip, fracx, fracy, fracz, isdeg = True):
    global_frac = FractureGlobal()
    global_frac.strike = strike
    global_frac.dip = dip
    global_frac.fracx = fracx
    global_frac.fracy = fracy
    global_frac.fracz = fracz
    global_frac.isdeg = isdeg
    global_frac.set_fracture_elements(frac.elements)
    return global_frac

class FractureZWellZ(FractureZ):
    """
    This class is a simplified solution for a monitor well that is parpendicular to the fracture plane
    Well path is along the Z direction
    Monitor well should be a WellZ class
    """

    def calculate_well_displacement(self):
        for well in self.monitor_wells:
            well.u3 = self.u3(well.x, well.y, well.z)
            well.calculate_strain()
    
class _ColorScales:
    white_red = [[0, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0, 0)']]
    white_green = [[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 255, 0)']]
    white_blue = [[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 0, 255)']]

class FractureGlobal(FractureZ):
    """
    This class calculate the displacement in the global coordinate system.
    Monitor well coordinates must be in the global coordinate system.
    Fracture strike and dip must be given. 
    """

    def __init__(self):
        super().__init__()
        self.strike = 90
        self.dip = 90
        self.fracx = 0
        self.fracy = 0
        self.fracz = 0
        self.isdeg = True
        self.global_monitor_wells = []
    

    def _well_global_to_local(self, well):
        """
        Convert global coordinates to local coordinates
        """
        strike = self.strike
        dip = self.dip
        local_coors = coor_global_to_local(well.x - self.fracx
                                           ,well.y - self.fracy
                                           ,well.z - self.fracz
                                           ,strike, dip, isdeg=self.isdeg)
        local_well = deepcopy(well)
        local_well.x = local_coors[0]
        local_well.y = local_coors[1]
        local_well.z = local_coors[2]
        return local_well
    
    def set_global_coors(self, strike, dip, fracx, fracy, fracz):
        self.fracx = fracx
        self.fracy = fracy
        self.fracz = fracz
        self.strike = strike
        self.dip = dip
        self.set_monitor_wells(self.global_monitor_wells)
    
    def set_monitor_wells(self, monitor_wells):
        if not isinstance(monitor_wells, list):
            monitor_wells = [monitor_wells]
        local_wells = []
        for well in monitor_wells:
            local_wells.append(self._well_global_to_local(well))

        self.monitor_wells = local_wells
        self.global_monitor_wells = deepcopy(monitor_wells)
        return self
    
    def draw_data(self, **kwargs):
        return self.draw_global_data(**kwargs)
    
    def draw_global_data_frac_only(self, tensile=None, S1=None, S2=None, **kwargs):
        obj_lst = []
        for elem in self.elements:
            x1,x2,x3 = elem.get_plot_coors()
            x,y,z = coor_local_to_global(x1, x2, x3, self.strike, self.dip, self.isdeg)
            x = x + self.fracx
            y = y + self.fracy
            z = z + self.fracz
            mesh_to_add = []
            if tensile is not None:
                c = np.abs(elem.width)
                mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5
                                 ,colorscale=_ColorScales.white_blue
                                 ,intensity=[c,c,c,c]
                                 ,cmin=0, cmax=tensile
                                 ,colorbar=dict(title='W',x=1))
                mesh_to_add.append(mesh)
            if S1 is not None:
                c = np.abs(elem.S1)
                mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5
                                 ,colorscale=_ColorScales.white_red
                                 ,intensity=[c,c,c,c]
                                 ,cmin=0, cmax=S1
                                 ,colorbar=dict(title='S1',x=1.1))
                mesh_to_add.append(mesh)
            if S2 is not None:
                c = np.abs(elem.S2)
                mesh = go.Mesh3d(x=x, y=y, z=z, opacity=0.5
                                 ,colorscale=_ColorScales.white_green
                                 ,intensity=[c,c,c,c]
                                 ,cmin=0, cmax=S2
                                 ,colorbar=dict(title='S2',x=1.2))
                mesh_to_add.append(mesh)
            if len(mesh_to_add) == 0:
                mesh = go.Mesh3d(x=x, y=y, z=z, color='blue', opacity=0.5)
                mesh_to_add.append(mesh)

            obj_lst += mesh_to_add
        return obj_lst

    def draw_global_data(self,**kwargs):

        obj_lst = self.draw_global_data_frac_only(**kwargs)

        for well in self.global_monitor_wells:
            obj_lst.append(well.draw_data())
        return obj_lst

    def calculate(self):
        self.calculate_well_displacement()
        for local_well, global_well in zip(
                self.monitor_wells, self.global_monitor_wells):
            global_well.strain = local_well.strain

    def plot_strain(self,**kwargs):
        for well in self.global_monitor_wells:
            well.plot_strain(**kwargs)

    def draw_local_data(self):
        obj_lst = []
        for elem in self.elements:
            x,y,z = elem.get_plot_coors()
            mesh = go.Mesh3d(x=x, y=y, z=z, color='blue', opacity=0.5)
            obj_lst.append(mesh)
        for well in self.monitor_wells:
            obj_lst.append(well.draw_data())
        return obj_lst

    def draw_local(self):
        fig = go.Figure(data=self.draw_local_data())
        fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        eye=dict(x=0, y=-1.5, z=0.5)
                    )
                )
            )
        return fig

    
    def draw(self,**kwargs):
        fig = go.Figure(data=self.draw_global_data(**kwargs))
        fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        eye=dict(x=0, y=-1.5, z=0.5)
                    )
                )
            )
        return fig

def _rotate_matrix(strike, dip):
    if dip == np.pi/2:
        dip = np.pi/2 - 1e-3
    strike = -strike + np.pi
    # dip = -dip
    rotate_mat = np.array([[1,0,0], [0, np.cos(dip), np.sin(dip)], [0, -np.sin(dip), np.cos(dip)]]) \
                @ np.array([[np.sin(strike), -np.cos(strike), 0], [np.cos(strike), np.sin(strike), 0], [0, 0, 1]])
    return rotate_mat

def coor_global_to_local(x, y, z, strike, dip,isdeg = True):
    """
    Convert global coordinates to local coordinates
    """
    if isdeg:
        strike = np.deg2rad(strike)
        dip = np.deg2rad(dip)
    global_coors = np.array([x, y, z])
    rotate_mat = _rotate_matrix(strike, dip)
    local_coors = rotate_mat @ global_coors
    return local_coors


def coor_local_to_global(x, y, z, strike, dip, isdeg = True):
    """
    Convert local coordinates to global coordinates
    """
    if isdeg:
        strike = np.deg2rad(strike)
        dip = np.deg2rad(dip)
    if dip == np.pi/2:
        dip = np.pi/2 - 1e-3
    local_coors = np.array([x, y, z])
    rotate_mat = _rotate_matrix(strike, dip)
    global_coors = np.linalg.inv(rotate_mat) @ local_coors
    return global_coors

def get_single_element_global_fracture(length, height, width, S1, S2, strike, dip, fracx, fracy, fracz):
    elem = Element.Element(length, height, width, S1, S2)
    elem.set_global_coors(0, 0, 0)
    frac = FractureGlobal()
    frac.set_global_coors(strike, dip, fracx, fracy, fracz)
    frac.set_fracture_elements(elem)
    return frac