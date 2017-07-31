# This program solves the 2D acoustic wave equation on a standard grid in first order form
# The purpose of this example is to gain familiarity with TensorFlow.
# The solution presented here is based on the TensorFlow PDE tutorial, found in tutorial/

# Governing equations:
# 
# p_t + u_x + v_y = 0
# u_t + p_x = 0
# v_t + p_y = 0

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import kernel as k

fig = plt.figure()

def display(a, rng=[0,1], show=True):
    if not show:
        return
    plt.cla()
    plt.imshow(a, vmin=rng[0], vmax=rng[1])
    plt.show()


def test():
    r = 2
    q = 2
    h = 1.0/(N-1)
    ux = np.linspace(0 + 0.5*h, 1 + 0.5*h, N)
    uy = np.linspace(0, 1, N)
    uX, uY = np.meshgrid(ux, uy)
    
    vx = np.linspace(0, 1, N)
    vy = np.linspace(0 + 0.5*h, 1 + 0.5*h, N)
    vX, vY = np.meshgrid(vx, vy)

    up  = taylor(uX, uY, q, r)
    udpx = taylor(uX, uY, q-1, r)

    #TODO: but visual inspection looks ok.

def pgrid(Nx, Ny):
    px = np.linspace(0, 1, Nx)
    py = np.linspace(0, 1, Ny)
    pX, pY = np.meshgrid(px, py)
    return pX, pY

sess = tf.InteractiveSession()

stride = 100
order = 4
N = 1024
nu = 0.1
nt = 2000
show_plot = False
s = k.stencils(order)
dof = 3*N**2

p0 = np.zeros([N, N], dtype=np.float32)
u0 = np.zeros([N, N], dtype=np.float32)
v0 = np.zeros([N, N], dtype=np.float32)

# Gaussian initial condition
x,y = pgrid(N, N)                               
a = 1000.0
p0 = np.exp(-a*np.power(x - 0.5,2) - a*np.power(y - 0.5,2)).astype(dtype=np.float32)

display(p0, rng=[-0.1, 0.1], show=show_plot)

p = tf.Variable(p0)
u = tf.Variable(u0)
v = tf.Variable(v0)

p_ = p - nu*(k.Dx2d(u, s) + k.Dy2d(v, s))
u_ = u - nu*k.Dxh2d(p, s)
v_ = v - nu*k.Dyh2d(p, s)

tf.global_variables_initializer().run()

step_pressure = tf.group(
  p.assign(p_))

step_velocity = tf.group(
  u.assign(u_), 
  v.assign(v_))

start = time.time()
for i in range(nt):
    step_pressure.run()
    step_velocity.run()
    if i % stride == 0:
        end = time.time()
        tn = (end - start)*1.0/stride
        
        print "Iteration: % d \t fps: %g \t s/iter: %g \t Gdof/s: %g  " % (i, 1/tn, tn, dof/tn*1e-9) 
        start = time.time()
        display(p.eval(), rng=[-0.1, 0.1], show=show_plot)
