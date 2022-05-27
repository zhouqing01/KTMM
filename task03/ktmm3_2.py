from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

# Create mesh and define function space
R = 1.0
circle_r = Circle(Point(0, 0), R)
mesh = generate_mesh(circle_r, 64)
V = FunctionSpace(mesh, 'P', 2)
bounds = MeshFunction("size_t", mesh, 1)
# -------------------ГРАНИЧНЫЕ УСЛОВИЯ----------------------------------
class boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[0] >= 0)
b1 = boundary1()
b1.mark(bounds, 0)
class boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[0] < 0)
b2 = boundary2()
b2.mark(bounds, 1)

# u 初值

T = 4.0
num_steps = 10
dt = T / num_steps
filenames = []
# alpha = 2
# beta = 3
t = 0
error_L2_ = []
error_C_ = []
time = []
# gamma = 10 #main formula
# u_D = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t',degree = 2, alpha = alpha, beta = beta, t = 0)
# f = Expression('beta - 2*gamma*(1+alpha)', degree=2, alpha = alpha, gamma = gamma, beta = beta)
# g = Expression('2*x[0]*x[0]/R + 2*alpha*x[1]*x[1]/R', degree=2, R = R, alpha = alpha, beta = beta)


# gamma = 2 #main formula
# u_D = Expression('3*x[1]*sin(x[0])*t', degree = 2, t = t)
#
# f = Expression('(1 + gamma*t) * 3*x[1]*sin(x[0])', degree = 2, gamma = gamma, t = t)
# g = Expression('-3*x[1]*cos(x[0])*x[0]*t/R - 3*sin(x[0])*x[1]*t/R', degree = 2, R = R, t = t)

#  u(x,y) = exp(y) + x*t
alpha = 1
u_D = Expression('exp(x[1]) + x[0]*t', degree = 2, t = 0) # == h
f = Expression('x[0] - alpha*exp(x[1])', degree = 2, alpha = alpha)
g = Expression('t*x[0]/R + exp(x[1])*x[1]/R', degree = 2, R = R, t = 0)

#  u(x,y) = 1+x^2+y^2+t
# alpha = 0.5
# u_D = Expression('1+x[0]*x[0]+x[1]*x[1]+t', degree=2, t=0)  # == h
# f = Expression('1 - 4*alpha', degree=2, alpha=alpha, t=0)
# g = Expression('2*x[0]*x[0]/R + 2*x[1]*x[1]/R', degree=2, R=R, t=0)


# u = t * r * sin(phi) + 1
# alpha = 1.0
# r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
# phi = Expression("atan2(x[1], x[0])", degree = 2)
# u_D = Expression("t * r * sin(phi) + 1", t = t, r = r, phi = phi, degree = 2)
# f = Expression("r * sin(phi)", t = t, r = r, phi = phi, degree = 2)
# g = Expression("t * sin(phi)", t = t, phi = phi, degree = 2)

bc = DirichletBC(V, u_D, bounds, 1)
u_i = interpolate(u_D, V)

# ----------------------ВАРИАЦИОННАЯ ЗАДАЧА----------------------------
u = TrialFunction(V)
v = TestFunction(V)
a = alpha * dt * dot(grad(u), grad(v)) * dx + u * v * dx
# L = (u_i + dt * f) * v * dx + alpha * dt * g * v * ds(0, subdomain_data=bounds)
L = (u_i + dt * f) * v * dx + alpha * dt * g * v * ds

n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
u = Function(V)
# t = 0
for n in range(num_steps):
    t += dt
    time.append(t)
    u_D.t = t
    solve(a == L, u, bc)
    u_e = interpolate(u_D, V)
    # u_e.t = t
    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    error_L2 = errornorm(u_e, u, 'L2')
    error_C = np.max(np.abs(vertex_values_u - vertex_values_u_e))
    print('t = ', t, ', error_L2 = ', error_L2, ', error_C = ', error_C)
    error_L2_.append(error_L2)
    error_C_.append(error_C)

    plt.figure()
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title('t = ' + "{:.1f}".format(t))
    plt.savefig('1_10/numerical' + "{:.1f}".format(t) + '.png')
    plt.figure()
    zfaces = np.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title('t = ' + "{:.1f}".format(t))
    plt.savefig('1_10/analytical' + "{:.1f}".format(t) + '.png')
    u_i.assign(u)

print(error_L2_)
print(error_C_)
plt.figure()
plt.title("Нормы")
plt.plot(time, error_L2_, 'r-', label='L2')
plt.plot(time, error_C_, 'b-', label='C')
plt.legend()
plt.savefig('norms.png')
plt.figure()
plt.plot(time, error_L2_, 'r-', time, error_L2_, 'ko', label='L2')
plt.legend()
plt.savefig('L_2_norm.png')
plt.figure()
plt.plot(time, error_C_, 'b-', time, error_C_, 'ko', label='L2')
plt.legend()
plt.savefig('C_norm.png')