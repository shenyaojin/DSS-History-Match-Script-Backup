import numpy as np
import matplotlib.pyplot as plt


class Element:

    def __init__(self, length, height, width, S1=0.0, S2=0.0):
        """
        # input width, S1, and S2 are reversed as defined in Yongzan's thesis
        # width = u3(0,0,0+) - u3(0,0,0-) 
        # S1 = u1(0,0,0+) - u1(0,0,0-)
        # S2 = u2(0,0,0+) - u2(0,0,0-)
        """
        self.height = height
        self.length = length
        self.width = -width
        self.S1 = -S1
        self.S2 = -S2
        self.mu = 0.3

    def set_coors(self,x1,x2,x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x3[x3==0] = 1e-16
    
    def set_global_coors(self, x1, x2, x3):
        self.abs_x1 = x1
        self.abs_x2 = x2
        self.abs_x3 = x3
    
    def get_plot_coors(self):
        x1 = np.array([self.abs_x1-self.length/2, self.abs_x1+self.length/2, self.abs_x1+self.length/2, self.abs_x1-self.length/2, self.abs_x1-self.length/2])
        x2 = np.array([self.abs_x2-self.height/2, self.abs_x2-self.height/2, self.abs_x2+self.height/2, self.abs_x2+self.height/2, self.abs_x2-self.height/2])
        x3 = np.array([self.abs_x3, self.abs_x3, self.abs_x3, self.abs_x3, self.abs_x3])
        return x1,x2,x3

    @property
    def u1(self):
        v = self.mu
        term1 = (2*(1-v)*self.J3()-self.x3*self.J4())*self.S1
        term2 = self.x3*self.J7()*self.S2
        term3 = ((1-2*v)*self.J1()+self.x3*self.J8())*self.width
        return 1/(8*np.pi*(1-v))*(term1-term2-term3)
    
    @property
    def u2(self):
        v = self.mu
        term1 = self.x3*self.J7()*self.S1
        term2 = (2*(1-v)*self.J3()-self.x3*self.J5())*self.S2
        term3 = ((1-2*v)*self.J2()+self.x3*self.J9())*self.width
        return 1/(8*np.pi*(1-v))*(-term1+term2-term3)
    
    @property
    def u3(self):
        v = self.mu
        term1 = ((1-2*v)*self.J1() - self.x3*self.J8())*self.S1
        term2 = ((1-2*v)*self.J2() - self.x3*self.J9())*self.S2
        term3 = (2*(1-v)*self.J3() - self.x3*self.J6())*self.width
        return 1/(8*np.pi*(1-v))*(term1+term2+term3)

    
    def r(self,xi1,xi2):
        return np.sqrt((xi1-self.x1)**2+(xi2-self.x2)**2+self.x3**2)
    
    def J1(self):
        def fun(xi1,xi2):
            return np.log(self.r(xi1,xi2)+self.x2-xi2)
        return self.chinnery_integral(fun)
    
    def J2(self):
        def fun(xi1,xi2):
            return np.log(self.r(xi1,xi2)+self.x1-xi1)
        return self.chinnery_integral(fun)

    def J3(self):
        def fun(xi1,xi2):
            return -np.arctan((self.x1-xi1)*(self.x2-xi2)/self.x3/self.r(xi1,xi2))
        return self.chinnery_integral(fun)
    
    def J4(self):
        def fun(xi1,xi2):
            return (self.x1-xi1)/self.r(xi1,xi2)/(self.r(xi1,xi2)+self.x2-xi2)
        return self.chinnery_integral(fun)
    
    def J5(self):
        def fun(xi1,xi2):
            return (self.x2-xi2)/self.r(xi1,xi2)/(self.r(xi1,xi2)+self.x1-xi1)
        return self.chinnery_integral(fun)
    
    def J6(self):
        def fun(xi1,xi2):
            r = self.r(xi1,xi2)
            x1 = self.x1
            x2 = self.x2
            x3 = self.x3
            return (x1-xi1)*(x2-xi2)*(x3**2+r**2)/(r*(x3**2+(x1-xi1)**2)*(x3**2+(x2-xi2)**2))
        return self.chinnery_integral(fun)
    
    def J7(self):
        def fun(xi1,xi2):
            return 1/self.r(xi1,xi2)
        return self.chinnery_integral(fun)

    def J8(self):
        def fun(xi1,xi2):
            return self.x3/self.r(xi1,xi2)/(self.r(xi1,xi2)+self.x2-xi2)
        return self.chinnery_integral(fun)
    
    def J9(self):
        def fun(xi1,xi2):
            return self.x3/self.r(xi1,xi2)/(self.r(xi1,xi2)+self.x1-xi1)
        return self.chinnery_integral(fun)
    


    def chinnery_integral(self,fun):
        a = self.length/2
        b = self.height/2
        return fun(a,b)-fun(a,-b)-fun(-a,b)+fun(-a,-b)





