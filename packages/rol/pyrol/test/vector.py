from array import array
import numpy
import dolfin

class vector(object):

    """Simple vector class implemented in Python to be encapsulated
       by a ROL::Vector"""

    def __init__(self,n):
        self.n=n
        # self.data = array('d',[0]*self.n)
        self.data = dolfin.Vector(dolfin.mpi_comm_world(), self.n)

    def __setitem__(self,i,value):
        self.data[i] = value

    def __getitem__(self,i):
        return self.data[i]

    def plus(self, vec):
        self.data += vec.data

    def scale(self, a):
        self.data *= a

    def norm(self):
        return self.data.norm("l2")

    def dot(self, vec):
        return self.data.inner(vec.data)

    def dimension(self):
        return self.data.local_size()

    def clone(self):
        x = vector(self.n)
        return x

    def __str__(self):
        return str(self.data)


if __name__ == '__main__':

    v = vector(1000)
    print("type(v) = {0}".format(type(v)))

    v[0]=1.0
    u = v.clone()

    attributes = dir(v)

    print("vector implemented methods:")
    print("dimension   - {0}".format("dimension"   in attributes))
    print("clone       - {0}".format("clone"       in attributes))
    print("scale       - {0}".format("scale"       in attributes))
    print("plus        - {0}".format("plus"        in attributes))
    print("norm        - {0}".format("norm"        in attributes))
    print("dot         - {0}".format("dot"         in attributes))
    print("__setitem__ - {0}".format("__setitem__" in attributes))
    print("__getitem__ - {0}".format("__getitem__" in attributes))


    print(v)
    print(u)
