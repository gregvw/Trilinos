import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pyrol
import numpy as np

if __name__ == '__main__':

    """Check basic linear algebra of NumPy array vector class"""

    n = 10
    xnp = np.random.randn(n)
    vcheck,output = pyrol.testVector(xnp)
    print("\nTest ROL::Vector linear algebra interface for a NumPy array")
    print(output)
