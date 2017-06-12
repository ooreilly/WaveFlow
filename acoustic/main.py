# This program solves the 2D acoustic wave equation on a standard grid in first order form
# The purpose of this example is to gain familiarity with TensorFlow.
# The solution presented here is based on the TensorFlow PDE tutorial, found in tutorial/

# Governing equations:
# 
# p_t + u_x + v_y = 0
# u_t + p_x = 0
# v_t + p_y = 0

# Kernels

def Dx(x, order=2):
  
  if order == 2:
    stencil =[-0.5, 0 , 0.5]

  kdx = make_kernel(stencil)
  return simple_conv(x, laplace_k)
