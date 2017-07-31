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

def pgrid(Nx, Ny, Nz):
    px = np.linspace(0, 1, Nx)
    py = np.linspace(0, 1, Ny)
    pz = np.linspace(0, 1, Nz)
    pX, pY, pZ = np.meshgrid(px, py, pz)
    return pX, pY, pZ

sess = tf.InteractiveSession()

stride = 200
order = 4
N = 100
nu = 0.01
nt = 2000
show_plot = False
s = k.stencils(order)
dof = 9*N**2

vx0 = np.zeros([N, N, N], dtype=np.float32)
vy0 = np.zeros([N, N, N], dtype=np.float32)
vz0 = np.zeros([N, N, N], dtype=np.float32)

sxx0 = np.zeros([N, N, N], dtype=np.float32)
sxy0 = np.zeros([N, N, N], dtype=np.float32)
sxz0 = np.zeros([N, N, N], dtype=np.float32)

syy0 = np.zeros([N, N, N], dtype=np.float32)
syz0 = np.zeros([N, N, N], dtype=np.float32)
szz0 = np.zeros([N, N, N], dtype=np.float32)

# Gaussian initial condition
x, y, z = pgrid(N, N, N)                               
a = 1000.0
vx0 = np.exp(-a*np.power(x - 0.5, 2) - a*np.power(y - 0.5, 2) -a*np.power(z - 0.5, 2)).astype(dtype=np.float32)
display(vx0[N/2,:,:], rng=[-0.1, 0.1], show=show_plot)

# Density
ax = 2.0
ay = 2.0
az = 2.0
# b: lame parameter, c: shear modulus
b  = 14.0
c  = 6.0
cxy = 6.0
cxz = 6.0
cyz = 6.0

vx = tf.Variable(vx0)
vy = tf.Variable(vy0)
vz = tf.Variable(vz0)

sxx = tf.Variable(sxx0)
sxy = tf.Variable(sxy0)
sxz = tf.Variable(sxz0)
syy = tf.Variable(syy0)
syz = tf.Variable(syz0)
szz = tf.Variable(szz0)

vx_x = k.Dx3d(vx, s)
vx_y = k.Dy3d(vx, s)
vx_z = k.Dz3d(vx, s)
vy_x = k.Dx3d(vy, s)
vy_y = k.Dy3d(vy, s)
vy_z = k.Dz3d(vy, s)
vz_x = k.Dx3d(vz, s)
vz_y = k.Dy3d(vz, s)
vz_z = k.Dz3d(vz, s)

sxx_x = k.Dxh3d(sxx, s)
sxy_x = k.Dxh3d(sxy, s)
sxz_x = k.Dxh3d(sxz, s)
sxy_y = k.Dyh3d(sxy, s)
syy_y = k.Dyh3d(syy, s)
syz_y = k.Dyh3d(syz, s)
sxz_z = k.Dzh3d(sxz, s)
syz_z = k.Dzh3d(syz, s)
szz_z = k.Dzh3d(szz, s)

vx_t = vx + nu*ax*(sxx_x + sxy_y + sxz_z)
vy_t = vy + nu*ay*(sxy_x + syy_y + syz_z)
vz_t = vz + nu*az*(sxz_x + syz_y + szz_z)

sxx_t = sxx + nu*( (b + 2*c)*vx_x + b*(vy_y + vz_z) )
syy_t = syy + nu*( (b + 2*c)*vy_y + b*(vx_x + vz_z) )
szz_t = szz + nu*( (b + 2*c)*vz_z + b*(vx_x + vy_y) )
sxy_t = sxy + nu*(  cxy*(vx_y + vy_x)               )
sxz_t = sxz + nu*(  cxz*(vx_z + vz_x)               )
syz_t = syz + nu*(  cyz*(vy_z + vz_y)               )

tf.global_variables_initializer().run()

step_stress = tf.group(
  sxx.assign(sxx_t), syy.assign(syy_t), szz.assign(szz_t), 
  sxy.assign(sxy_t), sxz.assign(sxz_t), syz.assign(syz_t)
  )

step_velocity = tf.group(
  vx.assign(vx_t), 
  vy.assign(vy_t), 
  vz.assign(vz_t), 
  )

start = time.time()
for i in range(nt):
    step_stress.run()
    step_velocity.run()
    if i % stride == 0:
        end = time.time()
        tn = (end - start)*1.0/stride
        print "Iteration: % d \t fps: %g \t s/iter: %g \t Gdof/s: %g  " % (i, 1/tn, tn, dof/tn*1e-9) 
        if show_plot:
            display(vx.eval()[N/2,:,:], rng=[-0.1, 0.1], show=show_plot)
        start = time.time()
