import pyrol
from vector import vector
from numpy.random import rand

if __name__ == '__main__':

    """Check basic linear algebra of vector class implemented in Python"""

    n = 1000
    x = vector(n)
    for i in range(x.dimension()):
      x[i] = rand()

    vcheck,output = pyrol.testVector(x)
#    print("vcheck = {0}".format(vcheck))
    print(output)
