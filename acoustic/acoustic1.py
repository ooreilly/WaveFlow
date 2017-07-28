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


def make_kernel_x(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1,1])
    return tf.constant(a, dtype=1)

def make_kernel_y(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape([1, len(a), 1,1])
    return tf.constant(a, dtype=1)

def Dx(x, stencil):
    return simple_conv(x, make_kernel_x(stencil))

def Dy(x, stencil):
    return simple_conv(x, make_kernel_x(stencil))

def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]

# Multivariate Taylor polynomial: x^q*y^r/(q!*r!)
def taylor(X, Y, q, r):
    return np.multiply(np.power(X, q), np.power(Y, r))

def stencils(orders):
    if order == 2:
        stencil = [-1, 1]
    if order == 4:
        stencil = [0.0416666666666667, -1.125, 1.125, -0.0416666666666667]
    return stencil

def test():
    # Build the different grids for each variable
    px = np.linspace(0, 1, N)
    py = np.linspace(0, 1, N)
    pX, pY = np.meshgrid(px, py)
    
    ux = np.linspace(0 + 0.5*h, 1 + 0.5*h, N)
    uy = np.linspace(0, 1, N)
    uX, uY = np.meshgrid(ux, uy)
    
    vx = np.linspace(0, 1, N)
    vy = np.linspace(0 + 0.5*h, 1 + 0.5*h, N)
    vX, vY = np.meshgrid(vx, vy)

    up  = taylor(uX, uY, q, r)
    udpx = taylor(uX, uY, q-1, r)


sess = tf.InteractiveSession()


stride = 100
order = 2
N = 4001
r = 2
q = 2
h = 1.0/(N-1)
dt = 0.5*h
nt = 1000
s = stencils(order)

#dpy = taylor(X, Y, q, r-1)

p0 = np.zeros([N, N], dtype=np.float32)
u0 = np.zeros([N, N], dtype=np.float32)
v0 = np.zeros([N, N], dtype=np.float32)

p = tf.Variable(p0)
u = tf.Variable(u0)
v = tf.Variable(v0)

p_ = p - dt*(Dx(u, s) + Dy(v, s))
u_ = u - dt*Dx(p, s)
v_ = v - dt*Dy(p, s)

tf.global_variables_initializer().run()

step = tf.group(
  p.assign(p_),
  u.assign(u_), 
  v.assign(v_))

start = time.time()
for i in range(nt):
    step.run()
    if i % stride == 0:
        end = time.time()
        print "Iteration: % d \t fps: %g \t iter/s: %g  " % (i, stride*1.0/(end - start), (end-start)*1.0/stride) 
        start = time.time()
