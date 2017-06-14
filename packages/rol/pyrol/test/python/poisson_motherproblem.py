from firedrake import UnitSquareMesh, FunctionSpace, Function, \
    Expression, TrialFunction, TestFunction, inner, grad, DirichletBC, \
    dx, Constant, solve, assemble, as_backend_type, File
from firedrake_vector import fd_vector
import pyrol

n = 16
mesh = UnitSquareMesh(n, n)

use_correct_riesz_and_inner = True

V = FunctionSpace(mesh, "Lagrange", 1)  # space for state variable
M = FunctionSpace(mesh, "DG", 0)  # space for control variable
beta = 1e-4
yd = Function(V)
yd.interpolate(Expression("(1.0/(2*pi*pi)) * sin(pi*x[0]) * sin(pi*x[1])",
               degree=3))
# uncomment this for a 'more difficult' target distribution
# yd.interpolate(Expression("(x[0] <= 0.5)*(x[1] <= 0.5)", degree=1))



def solve_state(u):
    y = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(y), grad(v)) * dx
    L = u * v * dx
    bc = DirichletBC(V, Constant(0.0), "on_boundary")
    y = Function(V)
    solve(a == L, y, bc)
    return y


def solve_adjoint(u, y):
    lam = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(lam), grad(v)) * dx
    L = -(y-yd) * v * dx
    bc = DirichletBC(V, Constant(0.0), "on_boundary")
    lam = Function(V)
    solve(a == L, lam, bc)
    return lam

class L2Inner(object):

    def __init__(self):
        self.A = assemble(TrialFunction(M)*TestFunction(M)*dx)
        self.bcs = [DirichletBC(M, Constant(0.0), "on_boundary")]

    def eval(self, _u, _v):
        y = _v.copy()
        # y.zero()
        self.A.mult(_u, y)
        return _v.inner(y)

    def riesz_map(self, derivative):
        rhs = Function(M, val=derivative.dat)
        res = Function(M)
        solve(self.A, res, rhs, bcs=self.bcs)
        # solve(self.A, res, rhs, bcs=self.bcs,
        #       solver_parameters={
        #           'ksp_monitor': False,
        #           'ksp_rtol': 1e-9, 'ksp_atol': 1e-10, 'ksp_stol': 1e-16,
        #           'ksp_type': 'cg', 'pc_type': 'hypre',
        #           'pc_hypre_type': 'boomeramg'
        #       })
        return res.vector()

state_file = File("state.pvd")
control_file = File("control.pvd")

class Objective(object):
    '''Subclass of ROL.Objective to define value and gradient for problem'''
    def __init__(self, inner_product):
        self.inner_product = inner_product
        self.u = Function(M)
        self.y = Function(V)

    def value(self, x, tol):
        u = self.u
        y = self.y
        return assemble(0.5 * (y-yd) * (y-yd) * dx + 0.5 * beta * u * u * dx)

    def gradient(self, g, x, tol):
        u = self.u
        y = self.y
        lam = solve_adjoint(u, y)
        v = TestFunction(M)
        L = beta * u * v * dx - lam * v * dx
        deriv = assemble(L)
        if self.inner_product is not None:
            grad = self.inner_product.riesz_map(deriv)
        else:
            grad = deriv
        g.scale(0)
        g.data += grad

    def update(self, x, flag, iteration):
        u = Function(M, val=x.data)
        self.u.assign(u)
        y = solve_state(self.u)
        self.y.assign(y)
        control_file.write(self.u)
        state_file.write(self.y)

solve_opt = { "Algorithm" : "Line Search",
              "Step" : {
                "Line Search" : {
                   "Descent Method" : {
                     "Type" : "Quasi-Newton Method"
                   }
                 }
              }
            }
# if use_correct_riesz_and_inner:
#     inner_product = L2Inner()
# else:
    # inner_product = None
inner_product = None

obj = Objective(inner_product)

u = Function(M)
opt = fd_vector(u.vector())

solve_output = pyrol.solveUnconstrained(obj,opt,solve_opt)
print solve_output
# print type(opt)
File("res.pvd").write(Function(M, val=opt.data))
