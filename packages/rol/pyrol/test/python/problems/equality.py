import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =  os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)
from vector import vector
import numpy as np

class Objective(object):

    def value(self,x,tol):
#        print("value")
        a = np.prod(x[:])
        b = np.exp(a)
        c = 1+x[0]**3+x[1]**3
        d = c**2
        return b - 0.5*d

    def gradient(self,g,x,tol):
#        print("grad")
        a = np.prod(x[:])
        b = np.exp(a)
        c = 1+x[0]**3+x[1]**3
        d = c**2
        for i in range(g.dimension()):
            g[i] = a*b/x[i]
        g[0] -= 3*c*x[0]**2
        g[1] -= 3*c*x[1]**2

    def hessVec(self,hv,v,x,tol):
#        print("hess")
        xarr = np.array(x[:])
        a = np.prod(xarr)
        b = np.exp(a)
        c = 1+x[0]**3+x[1]**3
        d = a/xarr
        e = np.array([v[i]*d[i] for i in range(x.dimension())])
        f = np.sum(e)
        h = (3*v[0]*x[0]**2+3*v[1]*x[1]**2)
        M = 1-np.identity(5)
        hv[:] = b*(np.dot(M,e)/xarr + f*d[:])
        hv[0] -= 6*c*v[0]*x[0]+3*h*x[0]**2
        hv[1] -= 6*c*v[1]*x[1]+3*h*x[1]**2

    def invHessVec(self,ihv,v,x,tol):
#        print("invhess")
        I = np.identity(5)
        H = np.zeros((5,5))
        for k in range(5):
            self.hessVec(H[:,k],I[:,k],x)
        ihv[:] = np.linalg.solve(H,v)


class EqualityConstraint(object):

    def value(self,c,x,tol):
#        print("value(eq)")
        c[0] = x.dot(x) - 10.0
        c[1] = x[1]*x[2] - 5.0*x[3]*x[4]
        c[2] = x[0]**3 + x[1]**3 + 1.0

    def applyJacobian(self,jv,v,x,tol):
#        print("applyjac(eq)")
        jv[0] = 2*x.dot(v)
        jv[1] = x[2]*v[1]+x[1]*v[2]-5.0*(x[4]*v[3]+x[3]*v[4])
        jv[2] = 3.0*(x[0]*x[0]*v[0] + x[1]*x[1]*v[1])

    def applyAdjointJacobian(self,ajv,v,x,tol):
#        print("applyadjjac(eq)")
        ajv[:] = 2.0*x[:]*v[0]
        ajv[0] += 3.0*x[0]**2*v[2]
        ajv[1] += x[2]*v[1]+3.0*x[1]**2*v[2]
        ajv[2] += x[1]*v[1]
        ajv[3] -= 5.0*x[4]*v[2]
        ajv[4] -= 5.0*x[3]*v[2]

    def applyAdjointHessian(self,ahuv,u,v,x,tol):
#        print("applyadjhess(eq)")
        ahuv[0] = 2.0*u[0]*v[0] +                 6.0*u[2]*x[0]*v[0]
        ahuv[1] = 2.0*u[0]*v[1] +     u[1]*v[2] + 6.0*u[2]*x[1]*v[1]
        ahuv[2] = 2.0*u[0]*v[2] +     u[1]*v[1]
        ahuv[3] = 2.0*u[0]*v[3] - 5.0*u[1]*v[4]
        ahuv[4] = 2.0*u[0]*v[4] - 5.0*u[1]*v[3]
