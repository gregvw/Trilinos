from math import exp
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =  os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 

class Objective(object):

    def value(self,x,tol):
        x1,x2,x3,x4,x5 = x
        return exp(x1*x2*x3*x4*x5) - 0.5 * pow( (pow(x1,3)+pow(x2,3)+1.0), 2) 

    def gradient(self,g,x,tol):
        x1,x2,x3,x4,x5 = x

        expxi = exp(x1*x2*x3*x4*x5);

        g[0] = x2*x3*x4*x5 * expxi - 3*pow(x1,2) * (pow(x1,3) + pow(x2,3) + 1)
        g[1] = x1*x3*x4*x5 * expxi - 3*pow(x2,2) * (pow(x1,3) + pow(x2,3) + 1)
        g[2] = x1*x2*x4*x5 * expxi
        g[3] = x1*x2*x3*x5 * expxi
        g[4] = x1*x2*x3*x4 * expxi
     
    def hessVec(self,hv,v,x,tol):
        x1,x2,x3,x4,x5 = x
        v1,v2,v3,v4,v5 = v

        expxi = exp(x1*x2*x3*x4*x5);

        hv[0] = ( pow(x2,2)*pow(x3,2)*pow(x4,2)*pow(x5,2)*expxi-9.0*pow(x1,4)-6.0*(pow(x1,3)+pow(x2,3)+1.0)*x1 ) * v1  + \
                ( x3*x4*x5*expxi+x2*pow(x3,2)*pow(x4,2)*pow(x5,2)*x1*expxi-9.0*pow(x2,2)*pow(x1,2) ) * v2 + \
                ( x2*x4*x5*expxi+pow(x2,2)*x3*pow(x4,2)*pow(x5,2)*x1*expxi ) * v3 + \
                ( x2*x3*x5*expxi+pow(x2,2)*pow(x3,2)*x4*pow(x5,2)*x1*expxi ) * v4 + \
                ( x2*x3*x4*expxi+pow(x2,2)*pow(x3,2)*pow(x4,2)*x5*x1*expxi ) * v5
  
        hv[1] = ( x3*x4*x5*expxi+x2*pow(x3,2)*pow(x4,2)*pow(x5,2)*x1*expxi-9.0*pow(x2,2)*pow(x1,2) ) * v1  + \
                ( pow(x1,2)*pow(x3,2)*pow(x4,2)*pow(x5,2)*expxi-9.0*pow(x2,4)-6.0*(pow(x1,3)+pow(x2,3)+1.0)*x2 ) * v2  + \
                ( x1*x4*x5*expxi+pow(x1,2)*x3*pow(x4,2)*pow(x5,2)*x2*expxi ) * v3  + \
                ( x1*x3*x5*expxi+pow(x1,2)*pow(x3,2)*x4*pow(x5,2)*x2*expxi ) * v4  + \
                ( x1*x3*x4*expxi+pow(x1,2)*pow(x3,2)*pow(x4,2)*x5*x2*expxi ) * v5
  
        hv[2] = ( x2*x4*x5*expxi+pow(x2,2)*x3*pow(x4,2)*pow(x5,2)*x1*expxi ) * v1  + \
                ( x1*x4*x5*expxi+pow(x1,2)*x3*pow(x4,2)*pow(x5,2)*x2*expxi ) * v2  + \
                ( pow(x1,2)*pow(x2,2)*pow(x4,2)*pow(x5,2)*expxi ) * v3  + \
                ( x1*x2*x5*expxi+pow(x1,2)*pow(x2,2)*x4*pow(x5,2)*x3*expxi ) * v4  + \
                ( x1*x2*x4*expxi+pow(x1,2)*pow(x2,2)*pow(x4,2)*x5*x3*expxi ) * v5
  
        hv[3] = ( x2*x3*x5*expxi+pow(x2,2)*pow(x3,2)*x4*pow(x5,2)*x1*expxi ) * v1  + \
                ( x1*x3*x5*expxi+pow(x1,2)*pow(x3,2)*x4*pow(x5,2)*x2*expxi ) * v2  + \
                ( x1*x2*x5*expxi+pow(x1,2)*pow(x2,2)*x4*pow(x5,2)*x3*expxi ) * v3  + \
                ( pow(x1,2)*pow(x2,2)*pow(x3,2)*pow(x5,2)*expxi ) * v4  + \
                ( x1*x2*x3*expxi+pow(x1,2)*pow(x2,2)*pow(x3,2)*x5*x4*expxi ) * v5
  
        hv[4] = ( x2*x3*x4*expxi+pow(x2,2)*pow(x3,2)*pow(x4,2)*x5*x1*expxi ) * v1  + \
                ( x1*x3*x4*expxi+pow(x1,2)*pow(x3,2)*pow(x4,2)*x5*x2*expxi ) * v2  + \
                ( x1*x2*x4*expxi+pow(x1,2)*pow(x2,2)*pow(x4,2)*x5*x3*expxi ) * v3  + \
                ( x1*x2*x3*expxi+pow(x1,2)*pow(x2,2)*pow(x3,2)*x5*x4*expxi ) * v4  + \
                ( pow(x1,2)*pow(x2,2)*pow(x3,2)*pow(x4,2)*expxi ) * v5



   
class EqualityConstraint(object):

    def value(self,c,x,tol):
        x1,x2,x3,x4,x5 = x

        c[0] = x1*x1+x2*x2+x3*x3+x4*x4+x5*x5 - 10.0;
        c[1] = x2*x3 - 5.0*x4*x5;
        c[2] = x1*x1*x1 + x2*x2*x2 + 1.0;


    def applyJacobian(self,jv,v,x,tol):
        x1,x2,x3,x4,x5 = x
        v1,v2,v3,v4,v5 = v
        jv[0] = 2.0*(x1*v1+x2*v2+x3*v3+x4*v4+x5*v5);
        jv[1] = x3*v2+x2*v3-5.0*x5*v4-5.0*x4*v5;
        jv[2] = 3.0*x1*x1*v1+3.0*x2*x2*v2;


    def applyAdjointJacobian(self,ajv,v,x,tol):
        x1,x2,x3,x4,x5 = x
        v1,v2,v3 = v

        ajv[0] = 2.0*x1*v1+3.0*x1*x1*v3;
        ajv[1] = 2.0*x2*v1+x3*v2+3.0*x2*x2*v3;
        ajv[2] = 2.0*x3*v1+x2*v2;
        ajv[3] = 2.0*x4*v1-5.0*x5*v2;
        ajv[4] = 2.0*x5*v1-5.0*x4*v2;



    def applyAdjointHessian(self,ahuv,u,v,x,tol):
        x1,x2 = x[:2]
        v1,v2,v3,v4,v5 = v
        u1,u2,u3 = u

        ahuv[0] = 2.0*u1*v1 +             6.0*u3*x1*v1;
        ahuv[1] = 2.0*u1*v2 +     u2*v3 + 6.0*u3*x2*v2;
        ahuv[2] = 2.0*u1*v3 +     u2*v2;
        ahuv[3] = 2.0*u1*v4 - 5.0*u2*v5;
        ahuv[4] = 2.0*u1*v5 - 5.0*u2*v4;
