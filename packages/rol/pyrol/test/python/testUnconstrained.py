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

    test_opt = { "Check Gradient"   : True,
                 "Check HessVec"    : True,
                 "Check HessSym"    : True,
                 "Check InvHessVec" : True,
                 "Order"            : 2,
                 "Steps"            : [0.1**i for i in range(7)]
               }
    

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

    # Perform checks on gradient, Hessian, etc
    test_output = pyrol.testObjective(obj,x,test_opt)
    print(test_output)


    # Solve the optimization problem with the given options
    solve_output = pyrol.solveUnconstrained(obj,x,solve_opt)
    print(solve_output)


