# from array import array
from math import sqrt
import sys
import copy

class vector(object):

    """Simple vector class implemented in Python to be encapsulated
       by a ROL::Vector"""

    def __init__(self,data):
        self.n = len(data)
        self.data = copy.deepcopy(data)

    def __setitem__(self,i,value):
        self.data[i] = value

    def __getitem__(self,i):
        return self.data[i]

    def dot(self,x):
        result = 0;
        for i in range(self.n):
            result += self.data[i]*x[i]
        return result

    def norm(self):
        result = 0;
        for d in self.data:
            result += d**2
        return sqrt(result)

    def plus(self,x):
        for i in range(self.n):
            self.data[i] = self.data[i] + x[i]

    def scale(self,alpha):
        for i in range(self.n):
            self.data[i] = alpha*self.data[i];

    def set(self,x):
        for i in range(self.n):
            self.data[i] = x[i]

    def zero(self):
        for i in range(self.n):
            self.data[i] = 0.0;

    def axpy(self,alpha,x):
        for i in range(self.n):
            self.data[i] = self.data[i] + alpha*x[i];

    def dimension(self):
        return self.n

    def clone(self):
        return vector([0.0]*self.n)

    def __str__(self):
        return str(self.data)





if __name__ == '__main__':

    v = vector([0.0]*10)
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
