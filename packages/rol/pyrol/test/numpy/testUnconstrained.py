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
 
    test_opt = { "Check Gradient"   : True,
                 "Check HessVec"    : True,
                 "Check HessSym"    : True,
                 "Check InvHessVec" : True,
                 "Order"            : 2,
                 "Steps"            : [0.1**i for i in range(7)]
               }
       
    solve_opt = { "Algorithm" : "Line Search",
                  "Return Iterates" : "true", 
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

#    v = np.random.randn(dim)
#    hv = np.zeros(dim)
#    ihhv = np.zeros(dim)
#    print("v = {0}".format(v))
#    obj.hessVec(hv,v,x)
#    print("hv = {0}".format(hv))
#    obj.invHessVec(ihhv,hv,x)
#    print("ihhv = {0}".format(ihhv))

    # Perform checks on gradient, Hessian, etc
    test_output = pyrol.testObjective(obj,x,test_opt)
    print(test_output)

    # Solve the optimization problem with the given options
    solve_output = pyrol.solveUnconstrained(obj,x,solve_opt)

    if len(solve_output)>1:
        print(solve_output[0])
        print("Optimization vectors\n")
        xv = lambda i: "".join(["x[",str(i),"]"])
        print("".join(["iter/element  "]+["{0:14}".format(xv(i)) for i in range(dim)]))
        print("-"*160)
        for i,line in enumerate(solve_output[1].split('\n')):
            if len(line)>0:
                print("".join(["{0:7}{1:7}".format(i,"")]+["{0:14}".format(l) for l in line.split()]))
    else:
        print(solve_output)
      

