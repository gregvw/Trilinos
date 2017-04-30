import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pyrol
import numpy as np
from problems import zakharov 

if __name__ == '__main__':

    print("Testing solution of unconstrained problem using NumPy array")

    dim = 10
    obj = zakharov.Objective(dim) 
    
    solve_opt = { "Algorithm" : "Line Search",
                  "Step" : {
                    "Line Search" : {
                       "Descent Method" : {
                         "Type" : "Newton-Krylov"
                       } 
                     } 
                  }
                }

    # Initial guess of solution
    x = np.ones(dim) 

    # Solve the optimization problem with the given options
    output,iterates = pyrol.solveUnconstrained(obj,x,solve_opt)
    print(output)
     


