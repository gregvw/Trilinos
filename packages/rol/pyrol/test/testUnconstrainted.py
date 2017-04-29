import pyrol
from vector import vector
from problems import zakharov 

if __name__ == '__main__':

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
    x = vector(dim)
    for i in range(dim):
        x[i] = 1.0

    # Solve the optimization problem with the given options
    output = pyrol.solveUnconstrained(obj,x,solve_opt)
    print(output)


