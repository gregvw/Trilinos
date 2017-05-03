import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =  os.path.dirname(parentdir)

sys.path.insert(0,grandparentdir)
from vector import vector

class Objective(object):

    def __init__(self,n):
#        print("zakharov::__init__")
        self.n = n
        self.k = vector(n)
        for i in range(n):
            self.k[i] = i+1.0

    def value(self,x,tol):
#        print("zakharov::value")
        xdotx = x.dot(x);
        kdotx = self.k.dot(x)
        val = xdotx + (kdotx**2)/4.0 + (kdotx**4)/16.0
        return val

    def gradient(self,g,x,tol):
#        print("zakharov::gradient")
        kdotx = self.k.dot(x)
        coeff = 0.25*(2.0*kdotx+kdotx**3)
        for i in range(self.n):
            g.data[i] = 2.0*x[i]+coeff*self.k[i]

    def dirDeriv(self,x,d,tol):
#        print("zakharov::derDeriv")
        kdotd = self.k.dot(d)
        kdotx = self.k.dot(x)
        xdotd = x.dot(d)
        coeff = 0.25*(2.0*kdotx+kdotx**3)
        deriv = 2*xdotd + coeff*kdotd
        return deriv

    def hessVec(self,hv,v,x,tol):
#        print("zakharov::hessVec")
        kdotx = self.k.dot(x)
        kdotv = self.k.dot(v)
        coeff = 0.25*(2.0+3.0*kdotx**2)*kdotv
        for i in range(self.n):
            hv[i] = 2.0*v[i]+coeff*self.k[i]

    def invHessVec(self,ihv,v,x,tol):
#        print("zakharov::invHessVec")
        kdotv  = self.k.dot(v)
        kdotx  = self.k.dot(x)
        kdotk  = self.k.dot(k)
        coeff  = -kdotv/(2.0*kdotk+16.0/(2.0+3.0*kdotx**3))
        for i in range(self.n):
            ihv[i] = 0.5*v[i]+coeff*self.k[i]
