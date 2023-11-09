'''
Solving Poisson equation with mixed boundary condition, example from
https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/mixed-poisson/python/documentation.html
solution file ell_mixed.csv generated using FENICS
'''
from infsupnet.dataset import  load_data
from infsupnet.equation import EllipticEQ, opA
from infsupnet.model import MODEL
from infsupnet.train import TrainLoop
import argparse
from infsupnet.utils import add_dict_to_argparser, args_to_dict
import torch 
torch.manual_seed(1234)
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import grad 
torch.set_default_dtype(torch.float64)
import matplotlib.tri as tri

default_dict = dict(
    d=2,
    num_int=int(10000),
    num_ext=int(200),
    hdim=400,
    depth=4,
    num_epoches=1000,
    device='cuda',
    batch_size=1000,
    folder = './result/result_mixedBC',
    lr=2e-4,
    optimizer='rmsprop',
    use_scheduler=False,
    StepLR=5000
)
df = pd.read_csv("ell_mixed.csv")
x_ = df['x'].values
y_ = df['y'].values
ur = df['u'].values
xx = np.zeros((len(x_),2),dtype=np.float64)
xx[:,0] = x_.astype(np.float64)
xx[:,1] = y_
x_test = xx



def compute_err(unet,ur,x_test):
    uu = unet(x_test).detach().cpu()[:,0]
    return np.linalg.norm(ur-uu.numpy())/np.linalg.norm(ur)




if __name__ == "__main__":
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

    g = lambda x: torch.sin(5*x[:,0])*((x==1)[:,1] + (x==0)[:,1])
    def f(x):
        return 10*torch.exp(-(((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2))/0.02)
    def opB(u,x):
        n = torch.zeros_like(x)
        n[(x==0)[:,0],0] = 0.
        n[(x==0)[:,1],1] = -1.
        n[(x==1)[:,0],0] = 0.
        n[(x==1)[:,1],1] = 1.
        u_x = grad(u, x,
                create_graph=True, retain_graph=True,
                grad_outputs=torch.ones_like(u),
                allow_unused=True)[0]
        return torch.sum(u_x * n,dim=-1)  + u[:,0] * ((x==0)[:,0] + (x==1)[:,0])

    x_test = torch.from_numpy(x_test).to(args['device'])

    eq = EllipticEQ(args['d'], f, g, opA, opB, ur=ur)
    a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=compute_err, x_test=x_test, device=args['device'],use_scheduler=args['use_scheduler'],stepLR=args['StepLR'])

    # a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=None, device=args['device'],use_scheduler=args['use_scheduler'])

    #a = TrainLoop(eq,unet,vnet,data,args['batch_size'],1e-4,1e-4,args['num_epoches'])
    a.run_loop()
    np.save(args['folder'] + "/Mixed_err_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.err)
    np.save(args['folder'] + "/Mixed_loss_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.losses)
    a.save_model(args['folder'] + "/Mixed_model_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']))

    # np.save(args['folder'] + "/err_%d"%args['d'],a.err)
    # np.save(args['folder'] + "/err_%d"%args['d'],a.losses)
    # a.save_model(args['folder']+ '/model_%d.pth'%args['d'])
