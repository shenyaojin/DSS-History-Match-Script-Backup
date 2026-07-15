
from . import Element, Fracture
from JIN_pylib import Data2D_XT
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class DynamicFracture:

    def __init__(self):
        """
        This is a class designed to capture dynamic fracture propagation.
        Critcal fields:
        self.fractures: a list of fractures
        self.monitor_wells: a list of monitor wells
        self.taxis: np.array of time coordinates, should have the same length as self.fractures
        """
        self.fractures = [] # a list of fractures
        self.monitor_wells = []  # a list of monitor wells
        self.taxis = None # np.array of time coordinates, should have the same length as self.fractures

    def set_fractures(self, fractures):
        """
        This function sets the fractures to the class
        fractures: a list of fractures
        """
        if not isinstance(fractures, list):
            fractures = [fractures]
        self.fractures = fractures
        return self
    
    def set_monitor_wells(self, monitor_wells):
        if not isinstance(monitor_wells, list):
            monitor_wells = [monitor_wells]
        self.monitor_wells = monitor_wells
        self._set_frac_monitor_wells()
    
    def _set_frac_monitor_wells(self):
        for frac in self.fractures:
            frac.set_monitor_wells(self.monitor_wells)
    
    def calculate(self):
        self.set_monitor_wells(self.monitor_wells)
        for frac in self.fractures:
            frac.calculate_well_displacement()
    
    def gather_strain_data(self):
        all_data = []
        for iwell in range(len(self.monitor_wells)):
            well_data = []
            for frac in self.fractures:
                well_data.append(frac.monitor_wells[iwell].strain)
            strain_data = Data2D_XT.Data2D()
            strain_data.data = np.array(well_data).T
            strain_data.taxis = self.taxis
            well = self.monitor_wells[iwell]
            strain_data.daxis = well.mds
            strain_data.x = well.x
            strain_data.y = well.y
            strain_data.z = well.z
            all_data.append(strain_data)
        return all_data
    
    def plot_width(self, plot_t = None, skip=10):
        if plot_t is None:
            plot_t = self.taxis[::skip]
        for t in plot_t:
            ind = np.argmin(np.abs(self.taxis-t))
            f = self.fractures[ind]
            x = [e.abs_x1 for e in f.elements]
            width = [-e.width for e in f.elements]
            plt.plot(x,width,label='{:.1f}'.format(self.taxis[ind]))

    def plot_height(self, plot_t = None, skip=10):
        if plot_t is None:
            plot_t = self.taxis[::skip]
        for t in plot_t:
            ind = np.argmin(np.abs(self.taxis-t))
            f = self.fractures[ind]
            x = [e.abs_x1 for e in f.elements]
            height = [e.height for e in f.elements]
            plt.plot(x,height,label='{:.1f}'.format(self.taxis[ind]))
        plt.legend()
    
    def draw(self,skip=5,**kwargs):
        frames = []
        for frac in self.fractures[::skip]:
            frames.append(go.Frame(data = frac.draw_data(**kwargs)))
        
        fig = go.Figure(
                data=frames[-1].data,
                layout=go.Layout(
                    updatemenus=[dict(type="buttons",
                                buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None])])]),
                frames=frames
            )
        
        fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    # xaxis=dict(range=[-50, 50]),  # xlim
                    # yaxis=dict(range=[-50, 50]),  # ylim
                    # zaxis=dict(range=[-50, 50]),  # ylim
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        eye=dict(x=0, y=-1.5, z=0.5)
                    )
                )
            )
        return fig
        
class GlobalDynamicFracture(DynamicFracture):

    def __init__(self):
        super().__init__()
        self.fracx = None
        self.fracy = None
        self.fracz = None
        self.strike = None
        self.dip = None
    
    def set_global_coors(self, strike, dip, fracx, fracy, fracz):
        self.fracx = fracx
        self.fracy = fracy
        self.fracz = fracz
        self.strike = strike
        self.dip = dip
    
    def set_monitor_wells(self, monitor_wells):
        if self.fracx is None:
            raise ValueError('Global coors not set yet')
        super().set_monitor_wells(monitor_wells)

    def convert_to_global(self):
        global_fracs = []
        for frac in self.fractures:
            global_frac = Fracture.fractureZ_to_global(frac, self.strike, self.dip
                                                    , self.fracx, self.fracy, self.fracz)
            global_fracs.append(global_frac)
        self.fractures = global_fracs

def _input_to_array(input_var, default_value, N):
    if input_var is None:
        return np.ones(N)*default_value
    elif isinstance(input_var, (int, float)):
        return np.ones(N)*input_var
    elif isinstance(input_var, (list, np.ndarray)):
        return input_var

class RectangularFracture(DynamicFracture):

    def define_LHW(self, taxis, length, heigth, width=None, S1=None, S2=None):
        self.taxis = taxis
        fractures = []
        width = _input_to_array(width, 0, len(taxis))
        S1 = _input_to_array(S1, 0, len(taxis))  # Assign default value to S1
        S2 = _input_to_array(S2, 0, len(taxis))  # Assign default value to S2
        for i in range(len(taxis)):
            elem = Element.Element(length[i], heigth[i], width[i], S1[i], S2[i])
            elem.set_global_coors(0, 0, 0)
            frac = Fracture.FractureZ().set_fracture_elements(elem)
            fractures.append(frac)
        self.fractures = fractures
        return self
    
    def define_asymmetric_grow(self, taxis, up, down, left, right
                              , width=None, S1 = None, S2 = None):
        up = _input_to_array(up, 0, len(taxis))
        down = _input_to_array(down, 0, len(taxis))
        left = _input_to_array(left, 0, len(taxis))
        right = _input_to_array(right, 0, len(taxis))
        width = _input_to_array(up, 0, len(taxis))
        S1 = _input_to_array(S1, 0, len(taxis))
        S2 = _input_to_array(S2, 0, len(taxis))
        self.taxis = taxis
        fractures = []
        for i in range(len(taxis)):
            L = left[i] + right[i]
            H = up[i] + down[i]
            X = (right[i] - left[i])/2
            Y = (up[i] - down[i])/2
            elem = Element.Element(L, H, width[i], S1[i], S2[i])
            elem.set_global_coors(X, Y, 0)
            frac = Fracture.FractureZ().set_fracture_elements(elem)
            fractures.append(frac)
        self.fractures = fractures

    def define_by_length(self, taxis, length, heigth, width, S1=0, S2=0):
        self.length_growth(taxis, length, heigth, width, S1, S2)
        return self

    def length_growth(self, taxis, length, heigth, width, S1=0, S2=0):
        """
        This function rectangular fracture changes its length with time
        """
        self.taxis = taxis
        fractures = []
        for i in range(len(taxis)):
            elem = Element.Element(length[i], heigth, width, S1, S2)
            elem.set_global_coors(0, 0, 0)
            frac = Fracture.FractureZ().set_fracture_elements(elem)
            fractures.append(frac)
        self.fractures = fractures
        return self

class GlobalRectangularFracture(RectangularFracture, GlobalDynamicFracture):

    def define_by_length(self, taxis, length, height, width, S1=0, S2=0):
        super().define_by_length(taxis, length, height, width, S1, S2)
        super().convert_to_global()
    
    def define_LHW(self, taxis, length, height, width=None, S1 = None, S2 = None):
        super().define_LHW(taxis, length, height, width, S1, S2)
        super().convert_to_global()
    
    def define_asymmetric_grow(self, taxis, up, down, left, right
                              , width=None, S1 = None, S2 = None):
        super().define_asymmetric_grow(taxis, up, down, left, right
                                      , width = width, S1 = S1, S2 = S2)
        super().convert_to_global()




class EllipticalFracture(DynamicFracture):

    def define_by_length(self,taxis, length, height_ratio, width_ratio, S1_ratio=0, S2_ratio=0, element_N=10):
        self.taxis = taxis
        fractures = []
        for i in range(len(taxis)):
            fracture_length = length[i]
            frac = Fracture.get_ellipse_fracture(fracture_length, height_ratio
                                                 , width_ratio, S1_ratio, S2_ratio, element_N)
            fractures.append(frac)
        self.fractures = fractures
        return self
    
    def define_by_length_asymmetric_updown(self, taxis, length, up_ratio, down_ratio, width_ratio, S1_ratio=0, S2_ratio=0, element_N=10):
        up_ratio = _input_to_array(up_ratio, 0, len(taxis))
        down_ratio = _input_to_array(down_ratio, 0, len(taxis))
        self.taxis = taxis
        fractures = []
        for i in range(len(taxis)):
            frac = Fracture.get_ellipse_asymmetric_updown(length[i], up_ratio[i]
                                                 , down_ratio[i], width_ratio, S1_ratio, S2_ratio, element_N)
            fractures.append(frac)
        self.fractures = fractures
        return self

class GlobalEllipticalFracture(EllipticalFracture, GlobalDynamicFracture):

    def define_by_length(self,taxis, length, height_ratio, width_ratio, S1_ratio=0, S2_ratio=0, element_N=10):
        super().define_by_length(taxis, length, height_ratio, width_ratio, S1_ratio, S2_ratio, element_N)
        super().convert_to_global()
    
    def define_by_length_asymmetric_updown(self, taxis, length, up_ratio, down_ratio, width_ratio, S1_ratio=0, S2_ratio=0, element_N=10):
        super().define_by_length_asymmetric_updown(taxis, length, up_ratio, down_ratio, width_ratio, S1_ratio, S2_ratio, element_N)
        super().convert_to_global()