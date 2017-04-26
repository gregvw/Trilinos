from __future__ import print_function
import os

if __name__ == '__main__':
    try:
        import numpy
        numpy_path = numpy.get_include()
        if os.path.exists(numpy_path):
            arrayobject_h = os.path.join(numpy_path,"numpy","arrayobject.h")
            if os.path.exists(arrayobject_h):
              print("{0};{1}".format(numpy_path,numpy.__version__))
        
    except:
        pass
