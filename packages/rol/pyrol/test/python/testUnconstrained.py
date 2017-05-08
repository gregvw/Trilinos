import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import pyrol
from vector import vector
from problems import zakharov

if __name__ == '__main__':

    print("Testing solution of unconstrained problem using Python vector class")

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
    x = vector([1.0]*dim)

    # Solve the optimization problem with the given options
    output = pyrol.solveUnconstrained(obj,x,solve_opt)
    print(output)
