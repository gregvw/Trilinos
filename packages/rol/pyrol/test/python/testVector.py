import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import pyrol
import random
from vector import vector

if __name__ == '__main__':

    """Check basic linear algebra of vector class implemented in Python"""

    n = 10

    x = vector(n)
    for i in range(n):
      x[i] = random.random()

    vcheck,output = pyrol.testVector(x)
    print("\nTest ROL::Vector linear algebra interface for a vector implemented in Python")
    print(output)
