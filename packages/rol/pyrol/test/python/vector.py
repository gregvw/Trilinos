from array import array
import numpy
import dolfin

dolfin.parameters['linear_algebra_backend'] = 'Eigen'

class vector(object):

    """Simple vector class implemented in Python to be encapsulated
       by a ROL::Vector"""

    def __init__(self, data):
        n = len(data)
        self.data = dolfin.Vector(dolfin.mpi_comm_world(), n)
        for (i, val) in enumerate(data):
            self.data[i] = val

    def __setitem__(self,i,value):
        if isinstance(i, slice):
            self.data[:] = value
        else:
            self.data[i] = value

    def __getitem__(self,i):
        if isinstance(i, slice):
            return self.data.array()
        else:
            return self.data[i][0]

    def plus(self, x):
        self.data += x.data

    def scale(self, a):
        self.data *= a

    def norm(self):
        return self.data.norm("l2")

    def dot(self, x):
        return self.data.inner(x.data)

    def axpy(self, a, x):
        self.data.axpy(a, x.data)

    def zero(self):
        self.data.zero()

    def set(self, x):
        self.data = x.data.copy()

    def dimension(self):
        return self.data.local_size()

    def clone(self):
        x = vector(self.data)
        return x

    def __str__(self):
        return str(self.data)


if __name__ == '__main__':

    v = vector([0]*10)
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

    print(u[:])
    print(u[1])

    print(v)
    print(u)
