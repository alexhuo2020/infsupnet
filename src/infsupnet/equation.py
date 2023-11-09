"""define the equation"""
import torch
from torch.autograd import grad
class EllipticEQ:
  """
  the class of infsupnet
  :param d: dimension of domain
  :param f: source function in the equation
  :param g: boundary data
  :param A: the elliptic operator
  :param B: the operator for the boundary condition

  """
  def __init__(self, d, f, g, opA, opB, ur=None):
    self.d = d
    self.f = f
    self.g = g
    self.opA = opA
    self.opB = opB
    self.ur = ur
def opA(u,x):
    d = len(x[0])
    #Compute Laplacian
    u_x = grad(u, x,
                    create_graph=True, retain_graph=True,
                    grad_outputs=torch.ones_like(u),
                    allow_unused=True)[0]
    u_xx = 0
    for i in range(d):
        u_xx += grad(u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx
def opB(u,x):
    return u[:,0]

if __name__=="__main__":
   f  = lambda x: 0
   g = lambda x: x
   eq = EllipticEQ(2,f,g,opA,opB)
