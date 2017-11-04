import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =  os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 
import numpy as np

class Objective(object):

    def value(self,x,tol):
        a = np.prod(x)
        b = np.exp(a) 
        c = 1+x[0]**3+x[1]**3
        d = c**2
        return b - 0.5*d

    def gradient(self,g,x,tol):
        a = np.prod(x)
        b = np.exp(a) 
        c = 1+x[0]**3+x[1]**3
        d = c**2
        g[:] = a*b/x[:]
        g[0] -= 3*c*x[0]**2 
        g[1] -= 3*c*x[1]**2

    def hessVec(self,hv,v,x,tol):
        a = np.prod(x)
        b = np.exp(a)
        c = 1+x[0]**3+x[1]**3
        d = a/x
        e = v*d
        f = np.sum(e)
        h = (3*v[0]*x[0]**2+3*v[1]*x[1]**2)
        M = 1-np.identity(5)

        hv[:] = b*(np.dot(M,e)/x+f*d[:])
        hv[0] -= 6*c*v[0]*x[0]+3*h*x[0]**2
        hv[1] -= 6*c*v[1]*x[1]+3*h*x[1]**2

#    def invHessVec(self,ihv,v,x,tol):
#        I = np.identity(5)
#        H = np.zeros((5,5))
#        for k in range(5):
#            self.hessVec(H[:,k],I[:,k],x)
#        ihv[:] = np.linalg.solve(H,v)


class EqualityConstraint(object):

    def value(self,c,x,tol):
        c[0] = x.dot(x) - 10.0
        c[1] = x[1]*x[2] - 5.0*x[3]*x[4]
        c[2] = x[0]**3 + x[1]**3 + 1.0    

    def applyJacobian(self,jv,v,x,tol):
        jv[0] = 2*x.dot(v)   
        jv[1] = x[2]*v[1]+x[1]*v[2]-5.0*(x[4]*v[3]+x[3]*v[4])
        jv[2] = 3.0*(x[0]*x[0]*v[0] + x[1]*x[1]*v[1])

    def applyAdjointJacobian(self,ajv,v,x,tol):
        ajv[0] = 2*x[0]*v[0] + 3.0*x[0]*x[0]*v[2]
        ajv[1] = 2*x[1]*v[0] + x[2]*v[1] + 3.0*x[1]*x[1]*v[2]
        ajv[2] = 2*x[2]*v[0] + x[1]*v[1]
        ajv[3] = 2*x[3]*v[0] - 5.0*x[4]*v[1]
        ajv[4] = 2*x[4]*v[0] - 5.0*x[3]*v[1]

    def applyAdjointHessian(self,ahuv,u,v,x,tol):
        ahuv[0] = 2.0*u[0]*v[0] +                 6.0*u[2]*x[0]*v[0]
        ahuv[1] = 2.0*u[0]*v[1] +     u[1]*v[2] + 6.0*u[2]*x[1]*v[1]
        ahuv[2] = 2.0*u[0]*v[2] +     u[1]*v[1]
        ahuv[3] = 2.0*u[0]*v[3] - 5.0*u[1]*v[4]
        ahuv[4] = 2.0*u[0]*v[4] - 5.0*u[1]*v[3]        
