import torch

def load_data(d, num_int, num_ext, batch_size, box= [-1,1], num_workers=0):
  """
  Mento carlo sampling on -1--1 
  :param d: dimension
  :param num_int: number of interior sampling points
  :param num_ext: number of exterior sampling points
  :param batch_size: batch size
  Return:
    two data loader, first for the interior points, second for the exterior points
  """

  x_dist = torch.distributions.Uniform(box[0], box[1])
  xs = x_dist.sample((num_int,d))
  xb = x_dist.sample((2*d,num_ext,d))
  for dd in range(d):
    xb[dd,:,dd] = torch.ones(num_ext)*box[1]
    xb[dd + d,:,dd] =  torch.ones(num_ext)*box[0]
  xb = xb.reshape(2*d*num_ext,d)
  dataloader_x = torch.utils.data.DataLoader(xs, batch_size=batch_size, shuffle=True,num_workers=num_workers)
  dataloader_xb = torch.utils.data.DataLoader(xb, batch_size=int(2*d*num_ext/(num_int/batch_size)), shuffle=True,num_workers=num_workers)
  print(xs.shape, xb.shape)
  assert len(xs)//batch_size == len(xb) // int(2*d*num_ext/(num_int/batch_size))
  return dataloader_x, dataloader_xb

def load_L_datand(d, num_int, num_ext, batch_size):
    base_dist0 = torch.distributions.uniform.Uniform(-1,1)
    base_dist1 = torch.distributions.uniform.Uniform(0,1)
    x = base_dist0.sample((2*num_int,d))
    ind = torch.prod(x > 0,-1)==0
    x = x[ind,:]
    # if len(ind) < num_int:
    #    x = x[ind,:]
    # else:
    #    x = x[:num_int,:]
    
    xb = base_dist0.sample((2*d,2*num_ext,d))
    for dd in range(d):
        xb[dd,:,dd] = torch.ones(2*num_ext)
        xb[dd + d,:,dd] =  -torch.ones(2*num_ext)
    xb = xb.reshape(2*d*2*num_ext,d)
    ind = torch.prod(xb>0,-1)==0
    xb = xb[ind,:]

    xb1 = base_dist1.sample((2*d,num_ext,d))
    for dd in range(d):
        xb1[dd,:,dd] = torch.ones(num_ext)
        xb1[dd + d,:,dd] =  torch.zeros(num_ext)
    xb1 = xb1.reshape(2*d*num_ext,d)
    ind = torch.prod(xb1>0,-1)==0
    xb1 = xb1[ind,:]
    xb = torch.concat([xb,xb1])
    idx =  torch.randperm(x.shape[0])
    x = x[idx].view(x.size())
    idx = torch.randperm(xb.shape[0])
    xb = xb[idx].view(xb.size())
    x = x[:batch_size* (num_int//batch_size),:]
    tmp = (len(x)//batch_size)
    xb = xb[:(num_ext//tmp)*tmp,:]
    # xb = xb[len(xb)//(len(x)//batch_size)]

    dataloader_x = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
    dataloader_xb = torch.utils.data.DataLoader(xb, batch_size=len(xb)//tmp, shuffle=True)
    # print(len(x),len(xb))
    # print(len(x) // batch_size)
    # print( (len(xb)//(len(x)//batch_size)))
    # print(len(xb)// (len(xb)//(len(x)//batch_size)))
    #print(len(x),len(xb))
    assert len(x)//batch_size ==  (len(xb)//(len(xb)//tmp))
    #return x, xb
    return dataloader_x, dataloader_xb




def load_L_data2d(num_int, num_ext, batch_size):
    """
    Mento Carlo sampling on the L-shape data 
    (-1,-1)->(1,-1)->(1,0)->(0,0)->(0,1)->(-1,1)->(-1,-1)
    Return:
      two dataset, first for interior points, second for boundary data points
    """
    d = 2
    base_dist1 = torch.distributions.uniform.Uniform(0,1)
    base_dist0 = torch.distributions.uniform.Uniform(-1,0)
    x = base_dist0.sample((int(num_int/3),d))
    xx1 = torch.cat([base_dist0.sample((int(num_int/3),1)),base_dist1.sample((int(num_int/3),1))],-1)
    xx2 = torch.cat([base_dist1.sample((int(num_int/3),1)),base_dist0.sample((int(num_int/3),1))],-1)
    x = torch.concat([x,xx1,xx2])
    bsize=num_ext
    x1 = torch.cat([base_dist0.sample((bsize,1)),torch.ones(bsize,1)],-1)
    x2 = torch.cat([base_dist0.sample((bsize,1)),-torch.ones(bsize,1)],-1)
    x3 = torch.cat([-torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
    x4 = torch.cat([-torch.ones(bsize,1),base_dist1.sample((bsize,1))],-1)
    x5 = torch.cat([torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
    x6 = torch.cat([torch.zeros(bsize,1),base_dist1.sample((bsize,1))],-1)
    x7 = torch.cat([base_dist1.sample((bsize,1)),-torch.ones(bsize,1)],-1)
    x8 = torch.cat([base_dist1.sample((bsize,1)),torch.zeros(bsize,1)],-1)
    xb = torch.concat([x1,x2,x3,x4,x5,x6,x7,x8])
    dataloader_x = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
    dataloader_xb = torch.utils.data.DataLoader(xb, batch_size=len(xb)//(len(x)//batch_size), shuffle=True)
    assert len(x)//batch_size == len(xb) // (len(xb)//(len(x)//batch_size))
    print(x.shape, xb.shape)
    return dataloader_x, dataloader_xb


def load_data_dfgbenchmark(num_int, num_ext):
  batch_size = num_int
  x_dist = torch.distributions.uniform.Uniform(0.,2.2)
  y_dist = torch.distributions.uniform.Uniform(0.,0.41)
  x1 = x_dist.sample((batch_size,1))
  x2 = y_dist.sample((batch_size,1))
  x = torch.concat([x1,x2],1)
  x=x[((x[:,0]-0.2)**2 + (x[:,1]-0.2)**2-0.05**2)>0]


  x1b = x_dist.sample((int(num_ext),1))
  x2b = y_dist.sample((int(num_ext),1))

  xbin = torch.concat([torch.zeros((int(num_ext),1)),x2b],1)
  xbout = torch.concat([2.2*torch.ones((int(num_ext),1)),x2b],1)

  xbud1 = torch.concat([x1b,torch.zeros((int(num_ext),1))],1)
  xbud2 = torch.concat([x1b,0.41*torch.ones((int(num_ext),1))],1)
  xbud = torch.concat([xbud1,xbud2])

  # plt.plot(xbin[:,0],xbin[:,1],'*')
  # plt.plot(xbout[:,0],xbin[:,1],'*')
  # plt.plot(xbud[:,0],xbud[:,1],'*')

  theta_dist = torch.distributions.uniform.Uniform(0.,2*torch.pi)
  theta = theta_dist.sample((50,1))
  xcircle = torch.concat([0.2+0.05*torch.cos(theta),0.2+0.05*torch.sin(theta)],1)
  xb0 = torch.concat([xbud,xcircle])
  return x, xb0, xbin, xbout



if __name__ == "__main__":
    xs, xb = load_data(3, 1024*10, 128*10, 128)
    # xs, xb = load_data(2, 10000, 100, 1000)
    print(len(list(iter(xs))))
    print(len(list(iter(xb))))

    xs, xb = load_L_data2d(10000,100,1000)
    print(len(list(iter(xs))))
    print(len(list(iter(xb))))
    xs, xb = load_L_datand(2,1024,128,128)
    print(len(list(iter(xs))))
    print(len(list(iter(xb))))

    # print(xs.shape)
    # print(xb.shape)
    import matplotlib.pyplot as plt 
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    #ax.scatter(xs[:,0],xs[:,1],xs[:,2])
    #ax.scatter(xb[:,0],xb[:,1],xb[:,2])
    plt.plot(xs[:,0],xs[:,1],'.')
    plt.plot(xb[:,0],xb[:,1],'.')
    plt.show()
