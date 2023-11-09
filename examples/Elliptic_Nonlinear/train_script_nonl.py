# Example from https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/nonlinear.html
# Solve equation -\nabla \cdot (q(u)\nabla u) = f, q(u) = (1+u)^m
# f=0, BC:
# u = 0 for x_1 = 0
# u = 1 for x_1 = 1
# \partial u/\partial n = 0 all other boundaries
# domain [0,1]^d
# Exact solution u(x_1,\ldots,x_d) = ((2^{m+1}-1)x_1+1)^{\frac{1}{m+1}}-1
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
    depth=4,
    num_epoches=10000,
    device='cuda',
    batch_size=1000,
    folder = './result/result_non',
    lr=2e-5,
    optimizer='rmsprop',
    use_scheduler=False
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
        u_xx += grad(a(u)*u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx
def opB(u,x):
    u_x = grad(u, x,
                create_graph=True, retain_graph=True,
                grad_outputs=torch.ones_like(u),
                allow_unused=True)[0]
    n = torch.zeros_like(x)
    for i in range(len(x[0])):
        n[(x==0)[:,i],i]=-1
        n[(x==1)[:,i],i]=1
    n[(x==0)[:,0],:]=0
    n[(x==1)[:,0],:]=0
    return u[:,0]*(x==0)[:,0] + (u[:,0]-1)*(x==1)[:,0] + torch.sum(n*u_x,dim=-1)

if __name__ == "__main__":
    m = 2
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser,default_dict)
    args = parser.parse_args()
    keys = default_dict.keys()
    args=args_to_dict(args,keys)
    data = load_data(d=args['d'], num_int=args['num_int'], num_ext=args['num_ext'], batch_size=args['batch_size'],box=[0,1])
    unet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    vnet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    if args['optimizer']=='adam':
        uop = torch.optim.Adam(unet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
        vop = torch.optim.Adam(vnet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
    elif args['optimizer']=='rmsprop':
        uop = torch.optim.RMSprop(unet.parameters(), lr=args['lr'])
        vop = torch.optim.RMSprop(vnet.parameters(), lr=args['lr'])

    def f(x):
        return 0
    ur = lambda x: ((2**(m+1)-1)*x[:,0]+1)**(1/(m+1)) - 1.
    g = lambda x: 0

    a_x = lambda x: (1 + x[:,0])**m
    opA = partial(opA,a=a_x)
    eq = EllipticEQ(args['d'], f, g, opA, opB, ur)
    x_test = torch.rand((10000,args['d']))
    a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=compute_err, x_test=x_test, device=args['device'],use_scheduler=args['use_scheduler'])
    a.run_loop()
    np.save(args['folder'] + "/Non_err_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.err)
    np.save(args['folder'] + "/Non_loss_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.losses)
    a.save_model(args['folder'] + "/Non_model_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']))
