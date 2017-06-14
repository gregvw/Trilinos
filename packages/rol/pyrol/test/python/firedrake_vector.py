from array import array
import numpy

class fd_vector(object):

    """Class to encapsulate a firedrake vector"""

    def __init__(self, data):
        self.data = data

    def __setitem__(self,i,value):
        self.data[i] = value

    def __getitem__(self,i):
        return self.data[i]

    def plus(self, x):
        self.data += x.data

    def scale(self, a):
        self.data *= a

    def norm(self):
        return self.data.inner(self.data)**0.5

    def dot(self, x):
        return self.data.inner(x.data)

    def axpy(self, a, x):
        self.data.axpy(a, x.data)

    def set(self, x):
        self.data = x.data.copy()

    def dimension(self):
        return len(self.data)

    def clone(self):
        return fd_vector(self.data.copy())
