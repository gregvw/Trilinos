import pyrol
import numpy as np

if __name__ == '__main__':

    """Check basic linear algebra of NumPy array vector class"""

    n = 10
    xnp = np.random.randn(n)
    vcheck,output = pyrol.testVector(xnp)
    print("\nTest ROL::Vector linear algebra interface for a NumPy array")
    print(output)
