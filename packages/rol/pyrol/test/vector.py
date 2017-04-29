from array import array
from math import sqrt
import sys

showcalls = True
#showcalls = False 
#if len(sys.argv)>1 :
#    if sys.argv[1] == "show calls":
#        showcalls = True

class vector(object):

    """Simple vector class implemented in Python to be encapsulated
       by a ROL::Vector"""


    def __init__(self,n):
        if showcalls:
            print("__init__")  
        self.n=n
        self.data = array('d',[0]*self.n)

    def __setitem__(self,i,value):
        self.data[i] = value

    def __getitem__(self,i):
        return self.data[i]

    def dot(self,x):
        if showcalls:
            print("dot")  
        result = 0;
        for i in range(self.n):
            result += self.data[i]*x[i]
        return result

#    def norm(self):
#        if showcalls:
#            print("norm")  
#        result = 0;
#        for i in range(self.n):
#            result += self.data[i]*self.data[i]
#        return sqrt(result)

    def plus(self,x):
        if showcalls:
            print("plus")  
        for i in range(self.n):
            self.data[i] = self.data[i] + x[i]

    def scale(self,alpha):
        if showcalls:
            print("plus")  
        for i in range(self.n):
            self.data[i] = alpha*self.data[i];

    def set(self,x):
        if showcalls:
            print("set")  
        for i in range(self.n):
            self.data[i] = x[i]

    def zero(self):
        if showcalls:
            print("zero")  
        for i in range(self.n):
            self.data[i] = 0.0; 
  
    def axpy(self,alpha,x): 
        if showcalls:
            print("axpy")  
        for i in range(self.n):
            self.data[i] = self.data[i] + alpha*x[i];

    def dimension(self):
        if showcalls:
            print("dimension")  
        return self.n

    def clone(self):
        if showcalls:
            print("clone")  
        x = vector(self.n)
        return x

    def __str__(self):
        return str(self.data)


if __name__ == '__main__':

    v = vector(10)
    print("type(v) = {0}".format(type(v)))

    v[0]=1.0
    u = v.clone()

    attributes = dir(v)

    print("vector implemented methods:")
    print("dimension   - {0}".format("dimension"   in attributes))
    print("clone       - {0}".format("clone"       in attributes))
    print("__setitem__ - {0}".format("__setitem__" in attributes))
    print("__getitem__ - {0}".format("__getitem__" in attributes))


    print(v)
    print(u)

    
