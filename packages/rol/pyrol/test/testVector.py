import pyrol
from vector import vector
import numpy as np

if __name__ == '__main__':

    """Check basic linear algebra of vector class implemented in Python"""



    n = 10
    x = vector(n)
    for i in range(n):
      x[i] = np.random.rand()

    vcheck,output = pyrol.testVector(x)
    print("\nTest ROL::Vector linear algebra interface for a vector implemented in Python")
    print(output)

    print('-'*100)
    xnp = np.random.randn(n)
    vcheck,output = pyrol.testVector(xnp)
    print("\nTest ROL::Vector linear algebra interface for a NumPy array")
    print(output)
