'''
Example from https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/fundamentals.html#tut-poisson-nd
Solve equation - \nabla \cdot (a(x) \nabla u(x)) = f in \Omega, u = u_0 on \partial\Omega
 u = 1 + |x|^2, a = \sum_{i=1}^d x_i
 f =- \sum_{j=1}^d \partial_{x_j} (a(x) 2 x_j) = -2d a(x) - 2\sum_j x_j = -(2d+2)\sum_i x_i
'''
from infsupnet.dataset import  load_data
from infsupnet.equation import EllipticEQ, opA, opB
from infsupnet.model import MODEL
from infsupnet.train import TrainLoop
import argparse
import torch 
torch.manual_seed(1234)
import torch.nn as nn
import numpy as np
from torch.autograd import grad 
from infsupnet.utils import compute_err
from infsupnet.utils import add_dict_to_argparser, args_to_dict
from functools import partial
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
default_dict = dict(
    d=2,
    num_int=int(10000),
    num_ext=int(100),
    hdim=40,
    depth=2,
    num_epoches=10000,
    device='cuda',
    batch_size=1000,
    folder = './result/',
    lr=2e-4,
    run='',
    use_scheduler=False,
)

def opA(u,x,a):
    d = len(x[0])
    #Compute Laplacian
    u_x = grad(u, x,
                    create_graph=True, retain_graph=True,
                    grad_outputs=torch.ones_like(u),
                    allow_unused=True)[0]
    u_xx = 0
    for i in range(d):
        u_xx += grad(a(x)*u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser,default_dict)
    args = parser.parse_args()
    keys = default_dict.keys()
    args=args_to_dict(args,keys)
    data = load_data(d=args['d'], num_int=args['num_int'], num_ext=args['num_ext'], batch_size=args['batch_size'])
    unet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    vnet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    if args['optimizer']=='adam':
        uop = torch.optim.Adam(unet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
        vop = torch.optim.Adam(vnet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
    elif args['optimizer']=='rmsprop':
        uop = torch.optim.RMSprop(unet.parameters(), lr=args['lr'])
        vop = torch.optim.RMSprop(vnet.parameters(), lr=args['lr'])

    def f(x):
        d = args['d']
        return  -(2*d+2)*torch.sum(x,dim=-1)#d*torch.pi**2/4*torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(d)]),0)
    ur = lambda x: 1 + torch.sum(x**2,dim=-1) 
    g = ur#lambda x: 0

    #ur = lambda x: 1 + torch.sum(x,dim=-1)# lambda x: torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range( args['d'])]),0)
    a_x = lambda x: torch.sum(x,dim=-1)
    opA = partial(opA,a=a_x)
    eq = EllipticEQ(args['d'], f, g, opA, opB, ur)
    x_test = torch.rand((10000,args['d']))
    a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=compute_err,x_test=x_test, device=args['device'],use_scheduler=args['use_scheduler'])
    #a = TrainLoop(eq,unet,vnet,data,args['batch_size'],args['lr'],args['lr'],args['num_epoches'],compute_err=compute_err, x_test=x_test)
    a.run_loop()
    np.save(args['folder'] + "/Semi_err_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.err)
    np.save(args['folder'] + "/Semi_loss_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.losses)
    a.save_model(args['folder'] + "/Semi_model_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']))
