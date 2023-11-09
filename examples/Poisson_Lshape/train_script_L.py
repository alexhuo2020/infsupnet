'''
Computation of the Poisson equation on L shaped domain, example from
https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/poisson/python/documentation.html#:~:text=The%20Poisson%20equation%20is%20the,gon%20%CE%93N.
The solution file ell.csv is generated using FENICS 
'''

from infsupnet.dataset import  load_L_data2d
from infsupnet.equation import EllipticEQ, opA, opB
from infsupnet.model import MODEL
from infsupnet.train import TrainLoop
import argparse
import torch 
torch.manual_seed(1234)
import torch.nn as nn
import numpy as np
from torch.autograd import grad 
from infsupnet.utils import add_dict_to_argparser, args_to_dict
import pandas as pd
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
import matplotlib.tri as tri

default_dict = dict(
    d=2,
    num_int=int(10000),
    num_ext=int(100),
    hdim=40,
    depth=4,
    num_epoches=20000,
    device='cuda',
    batch_size=1000,
    folder = './result/result_L/',
    lr=1e-4,
    optimizer='rmsprop',
    use_scheduler=True,
    StepLR=5000
)

# add function to process the result from FENICS
df = pd.read_csv("ell.csv")
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
    data = load_L_data2d(num_int=args['num_int'], num_ext=args['num_ext'], batch_size=args['batch_size'])
    unet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    vnet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh()).to(args['device'])
    if args['optimizer']=='adam':
        uop = torch.optim.Adam(unet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
        vop = torch.optim.Adam(vnet.parameters(), lr= args['lr'], betas=(0.5, 0.999),fused=True)
    elif args['optimizer']=='rmsprop':
        uop = torch.optim.RMSprop(unet.parameters(), lr=args['lr'])
        vop = torch.optim.RMSprop(vnet.parameters(), lr=args['lr'])

    def f(x):
        d = 2
        return torch.exp(-(x[:,0]+0.5)**2 - (x[:,1]+0.5)**2)#d*torch.pi**2/4*torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(d)]),0)
    g = lambda x: 0

    x_test = torch.from_numpy(x_test).to(args['device'])


    eq = EllipticEQ(args['d'], f, g, opA, opB, ur=ur)
    a = TrainLoop(eq,unet,vnet,data,uop,vop,args['num_epoches'],compute_err=compute_err, x_test=x_test, device=args['device'],use_scheduler=args['use_scheduler'],stepLR=args['StepLR'],Vol=3,Area=8)

    a.run_loop()
    np.save(args['folder'] + "/L_err_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.err)
    np.save(args['folder'] + "/L_loss_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']),a.losses)
    a.save_model(args['folder'] + "/L_model_%d_int%d_bd%d_hdim%d_depth%d_lr%.5f"%(args['d'],args['num_int'],args['num_ext'],args['hdim'],args['depth'],args['lr']))
