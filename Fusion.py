import numpy as np
import torch
from utils.tools import Warp


#Minimum-error fusion
def DispMEFusion(disps,sais,q):
    #disps:[4,h,w], sais:[81,3,h,w]
    img_ref=sais[40:41]
    img_ref=torch.from_numpy(img_ref).float()  #[1,3,h,w]
    disps=torch.from_numpy(disps).float() #[4,h,w]

    select_coordinates=[(1,1),(2,2),(3,3),(3,5),(2,6),(1,7),(5,5),(6,6),(7,7),(5,3),(6,2),(7,1),(3,4),(5,4),(4,3),(4,5)]
    num=len(select_coordinates)
    select_inds=[]
    for c in select_coordinates:
        select_inds.append(c[0]*9+c[1])

    source_imgs=sais[select_inds] #[N,3,h,w]
    source_imgs=torch.from_numpy(source_imgs).float() #[N,3,h,w]
    ref_imgs=img_ref.repeat(num,1,1,1) #[N,3,h,w]

    x_dist,y_dist=[],[]
    for c in select_coordinates:
        x_dist.append(4-c[1])
        y_dist.append(4-c[0])
    x_dist=torch.tensor(x_dist)
    y_dist=torch.tensor(y_dist)
    x_dist=x_dist.view(num,1,1,1) #[N,1,1,1]
    y_dist=y_dist.view(num,1,1,1) #[N,1,1,1]

    error_list=[]
    for disp in disps:
        disp=torch.unsqueeze(disp,dim=0) #[1,1,h,w]
        disp=disp.repeat(num,1,1,1) #[N,1,h,w]
        warp_imgs=Warp(source_imgs,disp,x_dist,y_dist) #[N,3,h,w]
        error=torch.mean(torch.abs(warp_imgs-ref_imgs),dim=1) #[N,h,w]
        error_std=torch.std(error,dim=0,keepdim=True) #[1,h,w]
        theta=torch.quantile(error_std, q)
        error_mean=torch.mean(error,dim=0,keepdim=True) #[1,h,w]
        error_median=torch.median(error,dim=0,keepdims=True)[0]
        error=(error_std>=theta).float()*error_median+(error_std<theta).float()*error_mean
        error_list.append(error)

    error=torch.cat(error_list,dim=0)  #[4,h,w]
    ind=torch.argmin(error, dim=0)  #[h,w]
    disp_fusion=torch.zeros_like(ind).float()
    for i in range(disps.size()[0]):
        disp_fusion+=disps[i]*(ind==i)
    disp_fusion=disp_fusion.numpy()

    return disp_fusion


#Weighed fusion
def DispWeightFusion(disps,sais,q,partial=True):
    #disps:[4,h,w], sais:[81,3,h,w]
    img_ref=sais[40:41]
    img_ref=torch.from_numpy(img_ref).float()  #[1,3,h,w]
    disps=torch.from_numpy(disps).float() #[4,h,w]

    #select_coordinates=[(1,1),(2,2),(3,3),(3,5),(2,6),(1,7),(5,5),(6,6),(7,7),(5,3),(6,2),(7,1),(3,4),(5,4),(4,3),(4,5)]
    select_coordinates=[(1,1),(2,2),(3,3),(3,5),(2,6),(1,7),(5,5),(6,6),(7,7),(5,3),(6,2),(7,1)]
    num=len(select_coordinates)
    select_inds=[]
    for c in select_coordinates:
        select_inds.append(c[0]*9+c[1])

    source_imgs=sais[select_inds] #[N,3,h,w]
    source_imgs=torch.from_numpy(source_imgs).float() #[N,3,h,w]
    ref_imgs=img_ref.repeat(num,1,1,1) #[N,3,h,w]

    x_dist,y_dist=[],[]
    for c in select_coordinates:
        x_dist.append(4-c[1])
        y_dist.append(4-c[0])
    x_dist=torch.tensor(x_dist)
    y_dist=torch.tensor(y_dist)
    x_dist=x_dist.view(num,1,1,1) #[N,1,1,1]
    y_dist=y_dist.view(num,1,1,1) #[N,1,1,1]

    error_list=[]
    for disp in disps:
        disp=torch.unsqueeze(disp,dim=0) #[1,1,h,w]
        disp=disp.repeat(num,1,1,1) #[N,1,h,w]
        warp_imgs=Warp(source_imgs,disp,x_dist,y_dist) #[N,3,h,w]
        error=torch.mean(torch.abs(warp_imgs-ref_imgs),dim=1) #[N,h,w]
        error_std=torch.std(error,dim=0,keepdim=True) #[1,h,w]
        theta=torch.quantile(error_std, q)

        error_mean=torch.mean(error,dim=0,keepdim=True) #[1,h,w]
        error_median=torch.median(error,dim=0,keepdims=True)[0]  #[1,h,w]

        occ_mask=(error_std>=theta).float()
        error=occ_mask*error_median+(1-occ_mask)*error_mean
        error_list.append(error)

    error=torch.cat(error_list,dim=0)  #[4,h,w]
    if partial:
        _,indices=torch.sort(error,dim=0)
        error_m1=torch.sum(error*(indices==0),dim=0,keepdim=True) #[1,h,w]
        error_m2=torch.sum(error*(indices==1),dim=0,keepdim=True) #[1,h,w]
        weight=torch.nn.Softmax(dim=0)(-torch.cat([error_m1,error_m2],dim=0)) #[2,h,w]
        disp_m1=torch.sum(disps*(indices==0),dim=0,keepdim=True) #[1,h,w]
        disp_m2=torch.sum(disps*(indices==1),dim=0,keepdim=True) #[1,h,w]
        disp_m12=torch.cat([disp_m1,disp_m2],dim=0) #[2,h,w]
        disp_fusion=torch.sum(disp_m12*weight,dim=0)
        disp_fusion=disp_fusion.numpy()
    else:
        weight=torch.nn.Softmax(dim=0)(-error)
        disp_fusion=torch.sum(disps*weight,dim=0)
        disp_fusion=disp_fusion.numpy()

    return disp_fusion
