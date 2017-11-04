import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =  os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 
import numpy as np

class Objective(object):

    def __init__(self,n):
#        print("zakharov::__init__")
        self.n = n
        self.k = np.array([1.0+i for i in range(self.n)])
        self.kdotk = sum([(i+1.0)**2 for i in range(self.n)])

    def value(self,x,tol=0):
#        print("zakharov::value")
        xdotx = x.dot(x);
        kdotx = self.k.dot(x)
        val = xdotx + (kdotx**2)/4.0 + (kdotx**4)/16.0
        return val

    def gradient(self,g,x,tol=0):
#        print("zakharov::gradient")
        kdotx = self.k.dot(x)
        coeff = 0.25*(2.0*kdotx+kdotx**3)
        g[:] = 2.0*x[:]+coeff*self.k[:]

    def dirDeriv(self,x,d,tol=0):
#        print("zakharov::derDeriv")
        kdotd = self.k.dot(d)
        kdotx = self.k.dot(x)
        xdotd = x.dot(d)
        coeff = 0.25*(2.0*kdotx+kdotx**3)
        deriv = 2*xdotd + coeff*kdotd
        return deriv

    def hessVec(self,hv,v,x,tol=0):
#        print("zakharov::hessVec")
        kdotx = self.k.dot(x)
        kdotv = self.k.dot(v)
        coeff = 0.25*(2.0+3.0*kdotx**2)*kdotv
        hv[:] = 2.0*v[:]+coeff*self.k[:]

    def invHessVec(self,ihv,v,x,tol=0):
#        print("zakharov::invHessVec")
        kdotv  = self.k.dot(v)
        kdotx  = self.k.dot(x)
        coeff  = -kdotv/(2.0*self.kdotk+16.0/(2.0+3.0*kdotx**2))
        ihv[:] = 0.5*v[:]+coeff*self.k[:]
        
        




