def make_kernel_x(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1,1])
    return tf.constant(a, dtype=1)

def make_kernel_y(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape([1, a.shape[0], 1,1])
    return tf.constant(a, dtype=1)

def Dx(x, stencil):
    return simple_conv(x, make_kernel_x(stencil))

def Dy(x, stencil):
    return simple_conv(x, make_kernel_y(stencil))

def Dxh(x, stencil):
    stencil = stencil + [0]
    return simple_conv(x, make_kernel_x(stencil))

def Dyh(x, stencil):
    stencil = stencil + [0]
    return simple_conv(x, make_kernel_y(stencil))

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
    if order == 6:
        stencil = [-3.0/640,  25.0/384,  -75.0/64,  75.0/64,  -25.0/384,  3/640]
    if order == 8:
        stencil = [0.5e1 / 0.7168e4, -0.49e2 / 0.5120e4, 0.245e3 / 0.3072e4, -0.1225e4 / 0.1024e4, 0.1225e4 / 0.1024e4,
                   -0.245e3 / 0.3072e4, 0.49e2 / 0.5120e4, -0.5e1 / 0.7168e4]
    return stencil
