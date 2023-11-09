'''
Solving Poisson equation on the domain [-1,1]^d with Dirichlet boundary conditions
'''
import os
import sys
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
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
torch.set_num_threads(8)
default_dict = dict(
    d=2,
    num_int=int(1000),
    num_ext=int(50),
    hdim=40,
    depth=2,
    num_epoches=20000,
    device='cuda',
    batch_size=1000,
    folder = './result',
    lr=1e-3,
    run='',
    use_scheduler=False,
    optimizer='rmsprop',
    StepLR=5000
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser,default_dict)
    args = parser.parse_args()
    keys = default_dict.keys()
    args=args_to_dict(args,keys)

    print("Solving the Poisson equation in dimension %d"%(args['d']))

    data = load_data(d=args['d'], num_int=args['num_int'], num_ext=args['num_ext'], batch_size=args['batch_size'])
    unet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    vnet = MODEL(args['d'], int(args['hdim']), args['depth'], act = nn.Tanh()).to(args['device'])
    if args['optimizer']=='adam':
        uop = torch.optim.Adam(unet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
        vop = torch.optim.Adam(vnet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
    elif args['optimizer']=='rmsprop':
        uop = torch.optim.RMSprop(unet.parameters(), lr=args['lr'])
        vop = torch.optim.RMSprop(vnet.parameters(), lr=args['lr'])

    def f(x):
        d = args['d']
        return d*torch.pi**2/4*torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(d)]),0)
    g = lambda x: 0

    ur = lambda x: torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range( args['d'])]),0)
    eq = EllipticEQ(args['d'], f, g, opA, opB, ur)
    x_test = 2*torch.rand((100000,args['d']))-1
    a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=compute_err, x_test=x_test, Area=2**(args['d']-1)*2*args['d'],Vol=2**args['d'],device=args['device'],use_scheduler=args['use_scheduler'],stepLR=args['StepLR'])
    a.run_loop()
    np.save(args['folder'] + "/err_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.err)
    np.save(args['folder'] + "/loss_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.losses)
    a.save_model(args['folder'] + "/model_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']))

