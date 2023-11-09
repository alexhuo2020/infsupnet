import torch
import torch.nn as nn

class TrainLoop:
    """
    Traing function
    :param infsupnet: the class of the model
    :param unet: network for the PDE solution
    :param vnet: network for the Lagrangian multiplier
    :param optim_u: the optimizer
    :param optim_u: the optimizer
    :dataloader_x: dataloader of interior sampling points
    :dataloader_xb: dataloader of boundary sampling points
    :Area: boundary surface area
    :Vol: volume of the domain
    :num_epoches: the number of epoches for training
    :device: 'cuda' is use gpu
    """

    def __init__(
        self,
        eq,
        unet,
        vnet,
        data,
        uop,
        vop,
        num_epochs,
        compute_err = None,
        x_test = None,
        Area = 1,
        Vol = 1,
        use_scheduler=False,
        stepLR = 5000,
        use_vop = True,
        device="cuda",
        pinn=False
   ):
        self.eq = eq
        self.unet = unet
        self.vnet = vnet
        self.data = data
        self.num_epochs = num_epochs
        self.compute_err = compute_err
        self.x_test = x_test
        self.Area = Area
        self.Vol = Vol
        self.losses = []
        self.err = []
        self.use_scheduler = use_scheduler
        self.stepLR = stepLR
        self.device = device# "cuda" if torch.cuda.is_available() else "cpu"
        self.optim_u = uop
        self.optim_v = vop
        self.pinn = pinn
    def run_loop(self):
        losses = []
        err = []
        if self.use_scheduler:
            scheduler_u = torch.optim.lr_scheduler.StepLR(self.optim_u,self.stepLR,0.5)
            scheduler_v = torch.optim.lr_scheduler.StepLR(self.optim_v,self.stepLR,0.5)
        for epoch in range(self.num_epochs):
            dataloader_x,dataloader_xb = self.data
            loss_epoch = 0 
            for ii, (x, xb) in enumerate(zip(dataloader_x,dataloader_xb)):
                x = x.to(self.device)
                xb = xb.to(self.device)
                x.requires_grad = True
                self.unet.zero_grad()
                u = self.unet(x)
                u_xx = self.eq.opA(u,x)
                v = self.vnet(x)[:,0]
                # v = self.eq.opA(v,x)
                xb.requires_grad = True
                ub = self.unet(xb)
                ubB = self.eq.opB(ub,xb)
                loss_u = (0.5*torch.mean((ubB-self.eq.g(xb))**2)*self.Area + torch.mean((- u_xx - self.eq.f(x))*v.detach())*self.Vol)
                if self.pinn:
                    loss_u = (0.5*torch.mean((ubB-self.eq.g(xb))**2)*self.Area + torch.mean((- u_xx - self.eq.f(x))**2)*self.Vol)
                self.losses.append(loss_u.item())
                loss_u.backward()
                self.optim_u.step()

                self.vnet.zero_grad()
                u = self.unet(x)
                v = self.vnet(x)[:,0]
                # v = self.eq.opA(v,x)
                u_xx = self.eq.opA(u,x) 
                loss_v = -torch.mean((-u_xx.detach() - self.eq.f(x))*v)*self.Vol
                loss_v.backward()
                self.optim_v.step()
                loss_epoch += loss_u.item()
            if self.compute_err is not None:
                err.append(self.compute_err(self.unet, self.eq.ur, self.x_test.to(self.device)).item())
            losses.append(loss_epoch/(ii+1))
            if self.use_scheduler:
                scheduler_u.step()
                scheduler_v.step()

            if epoch%100==0 and self.compute_err is not None:
                print('epoch: %d, loss: %f, err: %f'%(epoch,loss_epoch/(ii+1),err[-1]))
            elif epoch%100==0:
                print('epoch: %d, loss:%f'%(epoch,loss_epoch/(ii+1)))
        self.losses = losses
        self.err = err
    def save_model(self, filename):
        torch.save(self.unet.state_dict(), filename + 'unet.pth')
        torch.save(self.vnet.state_dict(), filename + 'vnet.pth')
    def load_model(self,filename):
        checkpoint = torch.load(filename + 'unet.pth')
        self.unet.load_state_dict(checkpoint)
        checkpoint = torch.load(filename + 'vnet.pth')
        self.vnet.load_state_dict(checkpoint)
