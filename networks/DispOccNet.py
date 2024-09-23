import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import ResBlock, ResBlock3D, UpSkip, UpSkip3D
sys.path.append('..')
from utils.tools import Warp

#Feature extractor
class Feature_extractor(nn.Module):
    def __init__(self,in_channels,base_channels,ASPP):
        super(Feature_extractor,self).__init__()
        num_channels=[base_channels,base_channels*2,base_channels*4]
        self.conv=nn.Sequential(nn.Conv2d(in_channels,num_channels[0],kernel_size=3,stride=1,padding=1),
                                nn.GroupNorm(num_groups=8, num_channels=base_channels),
                                nn.LeakyReLU(inplace=True))
        self.block1=ResBlock(num_channels[0],num_channels[1],stride=2)
        self.block2=ResBlock(num_channels[1],num_channels[2],stride=2)
        if ASPP:
            self.atrous1=nn.Sequential(nn.Conv2d(num_channels[2],num_channels[2],kernel_size=3,stride=1,padding=3,dilation=3),
                                       nn.GroupNorm(num_groups=8, num_channels=num_channels[2]),
                                       nn.LeakyReLU(inplace=True))
            self.atrous2=nn.Sequential(nn.Conv2d(num_channels[2],num_channels[2],kernel_size=3,stride=1,padding=6,dilation=6),
                                       nn.GroupNorm(num_groups=8, num_channels=num_channels[2]),
                                       nn.LeakyReLU(inplace=True))
            self.atrous3=nn.Sequential(nn.Conv2d(num_channels[2],num_channels[2],kernel_size=3,stride=1,padding=8,dilation=8),
                                       nn.GroupNorm(num_groups=8, num_channels=num_channels[2]),
                                       nn.LeakyReLU(inplace=True))
            self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(num_channels[2],num_channels[2],kernel_size=1,stride=1),
                                             nn.GroupNorm(num_groups=8, num_channels=num_channels[2]),
                                             nn.LeakyReLU(inplace=True))
        self.up1=UpSkip(num_channels[2],num_channels[1],mode='cat')                                     
        self.block3=ResBlock(num_channels[2],num_channels[1],stride=1)
        self.up2=UpSkip(num_channels[1],num_channels[0],mode='cat')
        self.block4=ResBlock(num_channels[1],num_channels[0],stride=1)
        self.aspp=ASPP
        
    def forward(self,x):
        map1=self.conv(x)  #[B,C,H,W]
        map2=self.block1(map1)  #[B,2C,H/2,W/2]
        map3=self.block2(map2)  #[B,4C,H/4,W/4]
        if self.aspp:
            a1=self.atrous1(map3)
            a2=self.atrous2(map3)
            a3=self.atrous3(map3)
            g=self.global_pool(map3)
            map3=a1+a2+a3+g
        map2=self.up1(map3,map2)  #[B,4C,H/2,W/2]
        map2=self.block3(map2)  #[B,2C,H/2,W/2]
        map1=self.up2(map2,map1)  #[B,2C,H,W]
        out=self.block4(map1)  #[B,C,H,W]
        return out


#Cost volume of stage 1
class Cost_volume_s1(nn.Module):
    def __init__(self,disp_sample,group_num):  
        super(Cost_volume_s1,self).__init__()
        self.disp_sample=disp_sample
        self.group_num=group_num

    def forward(self,feat_ref,feat_ls,feat_rs):
        D=len(self.disp_sample)
        B,C,H,W=feat_ref.size()
        cost=Variable(torch.FloatTensor(B,C,D,H,W).zero_()).to(feat_ref.device)
        k=0
        for i in self.disp_sample:
            if i!= 0 :
                warp_left=Warp(feat_ls,disp=i,x_dist=1,y_dist=0)
                warp_right=Warp(feat_rs,disp=i,x_dist=-1,y_dist=0)
            else:
                warp_left=feat_ls
                warp_right=feat_rs
                
            mu=(warp_left+warp_right+feat_ref)/3
            var=(torch.pow(warp_left-mu,2)+torch.pow(warp_right-mu,2)+torch.pow(feat_ref-mu,2))/3  
            cost[:,:,k,:,:]=var
            k+=1
            
        cost=cost.contiguous() #[B,C,D,H,W]
        assert C % self.group_num == 0
        channels_per_group = C // self.group_num
        cost = cost.reshape(B, self.group_num, channels_per_group, D, H, W).mean(dim=2)  #[B,G,D,H,W]
        
        return cost
    
#Cost volume of stage 2
class Cost_volume_s2(nn.Module):
    def __init__(self,res_sample,group_num):  
        super(Cost_volume_s2,self).__init__()
        self.res_sample=res_sample
        self.group_num=group_num
        
    def forward(self,feat_ref,feat_ls,feat_rs,disp_init):
        D=len(self.res_sample)
        B,C,H,W=feat_ref.size()
        cost=Variable(torch.FloatTensor(B,C,D,H,W).zero_()).to(feat_ref.device)
        k=0
        for res in self.res_sample:
            disp=disp_init+res
            warp_left=Warp(feat_ls,disp=disp,x_dist=1,y_dist=0)
            warp_right=Warp(feat_rs,disp=disp,x_dist=-1,y_dist=0)
                
            mu=(warp_left+warp_right+feat_ref)/3
            var=(torch.pow(warp_left-mu,2)+torch.pow(warp_right-mu,2)+torch.pow(feat_ref-mu,2))/3
            cost[:,:,k,:,:]=var
            k+=1
            
        cost=cost.contiguous()  #[B,C,D,H,W]
        assert C % self.group_num == 0
        channels_per_group = C // self.group_num
        cost = cost.reshape(B, self.group_num, channels_per_group, D, H, W).mean(dim=2)  #[B,G,D,H,W]
        
        return cost
    

#Cost filter
class Cost_filter(nn.Module):
    def __init__(self,in_channels):
        super(Cost_filter,self).__init__()
        self.block1=ResBlock3D(in_channels,in_channels,stride=2)
        self.block2=ResBlock3D(in_channels,in_channels*2,stride=2)
        self.block3=ResBlock3D(in_channels*2,in_channels*4,stride=2)
        
        self.up1=UpSkip3D(in_channels*4,in_channels*2,mode='sum')
        self.up_block1=ResBlock3D(in_channels*2,in_channels*2,stride=1)
        
        self.up2=UpSkip3D(in_channels*2,in_channels,mode='sum')
        self.up_block2=ResBlock3D(in_channels,in_channels,stride=1)
        
        self.up3=UpSkip3D(in_channels,in_channels,mode='sum')
        self.up_block3=ResBlock3D(in_channels,in_channels,stride=1)
        
    def forward(self,inp):
        map1=inp  #[B,G,D,H,W]
        map2=self.block1(map1)  #[B,G,D,H/2,W/2]
        map3=self.block2(map2)  #[B,2G,D,H/4,W/4]
        map4=self.block3(map3)  #[B,4G,D,H/8,W/8]
        
        map3=self.up1(map4,map3) #[B,2G,D,H/4,W/4]
        map3=self.up_block1(map3) #[B,2G,D,H/4,W/4]
        
        map2=self.up2(map3,map2) #[B,G,D,H/2,W/2]
        map2=self.up_block2(map2) #[B,G,D,H/2,W/2]
        
        map1=self.up3(map2,map1) #[B,G,D,H,W]
        out=self.up_block3(map1) #[B,G,D,H,W]
        
        return out


#Regression of stage 1
class Regression_s1(nn.Module):
    def __init__(self,in_channels,disp_sample):
        super(Regression_s1,self).__init__()
        self.conv=nn.Sequential(nn.Conv3d(in_channels,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                                nn.LeakyReLU(inplace=True))
        self.disp_sample=disp_sample
        
    def forward(self,x):
        x=self.conv(x)  #[B,1,D,H,W]
        x=torch.squeeze(x,dim=1)  #[B,D,H,W]
        
        prob=F.softmax(x,dim=1)
        disp_sample=torch.Tensor(np.reshape(self.disp_sample,[1,self.disp_sample.shape[0],1,1])).to(x.device)
        disp = torch.sum(prob*disp_sample.data, dim=1, keepdim=True)  #[B,1,H,W]
        
        return disp


#Regression of stage 2   
class Regression_s2(nn.Module):
    def __init__(self,in_channels,res_sample):
        super(Regression_s2,self).__init__()
        self.conv=nn.Sequential(nn.Conv3d(in_channels,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                                nn.LeakyReLU(inplace=True))
        self.res_sample=res_sample
        
    def forward(self,x):
        x=self.conv(x)  #[B,1,D,H,W]
        x=torch.squeeze(x,dim=1)  #[B,D,H,W]
        
        prob=F.softmax(x,dim=1)
        res_sample=torch.Tensor(np.reshape(self.res_sample,[1,self.res_sample.shape[0],1,1])).to(x.device)
        res = torch.sum(prob*res_sample.data, dim=1, keepdim=True)  #[B,1,H,W]
        
        return res
        

#Disparity estimation network
class DispNet(nn.Module):
    def __init__(self,in_channels,base_channels,filter_num,disp_sample,res_sample,group_num):
        super(DispNet,self).__init__()
        
        self.FE=Feature_extractor(in_channels,base_channels,ASPP=True)
            
        self.cost_volume_s1=Cost_volume_s1(disp_sample,group_num)
        self.cost_volume_s2=Cost_volume_s2(res_sample,group_num)
        
        filter_list=[]
        for i in range(filter_num):
            cost_filter=Cost_filter(group_num)
            filter_list.append(cost_filter)
        self.filter_list=nn.ModuleList(filter_list)
        
        self.regression_s1=Regression_s1(group_num,disp_sample)
        self.regression_s2=Regression_s2(group_num,res_sample)
        
    def forward(self,img_ls,img_ref,img_rs):
        B=img_ls.shape[0]
        img=torch.cat([img_ls,img_ref,img_rs],dim=0)  #[3B,3,H,W]
        feat=self.FE(img)  #[3B,C,H,W]
        feat_ls,feat_ref,feat_rs=feat[:B],feat[B:2*B],feat[2*B:]
        cost_s1=self.cost_volume_s1(feat_ref,feat_ls,feat_rs)  #[B,G,D,H,W]
    
        for cost_filter in self.filter_list:
            cost_s1=cost_filter(cost_s1)   #[B,G,D,H,W]

        disp_init=self.regression_s1(cost_s1) #[B,1,H,W]
        
        cost_s2=self.cost_volume_s2(feat_ref,feat_ls,feat_rs,disp_init)  #[B,G,D,H,W]
        
        for cost_filter in self.filter_list:
            cost_s2=cost_filter(cost_s2)   #[B,G,D,H,W]
        
        res=self.regression_s2(cost_s2)   #[B,1,H,W]
        disp=disp_init+res    #[B,1,H,W]
        
        return disp_init,disp


# Occlusion prediction network
class OccNet(nn.Module):
    def __init__(self,in_channels,base_channels):
        super(OccNet,self).__init__()
        num_channels=[base_channels,base_channels*2,base_channels*4,base_channels*8]
        self.conv1=nn.Sequential(nn.Conv2d(in_channels,num_channels[0],kernel_size=3,stride=1,padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=num_channels[0]),
                                 nn.LeakyReLU(inplace=True))
        self.block1=ResBlock(num_channels[0],num_channels[1],stride=2)
        self.block2=ResBlock(num_channels[1],num_channels[2],stride=2)
        
        self.up1=UpConcat(num_channels[2],num_channels[1])
        self.up_block1=ResBlock(num_channels[2],num_channels[1],stride=1)
        self.up2=UpConcat(num_channels[1],num_channels[0])
        self.up_block2=ResBlock(num_channels[1],num_channels[0],stride=1)
        
        self.conv2=nn.Sequential(nn.Conv2d(num_channels[0],2,kernel_size=3,stride=1,padding=1),
                                 nn.Softmax(dim=1))
        
    def forward(self,warp_left,warp_right,disp):
        inp=torch.cat([warp_left,warp_right,disp],1)  #[N,3*2+1,h,w]
        map1=self.conv1(inp)   #[N,16,h,w]
        map2=self.block1(map1)  #[N,32,h/2,w/2]
        map3=self.block2(map2)  #[N,64,h/4,w/4]
        map2=self.up1(map3,map2)  #[N,64,h/2,w/2]
        map2=self.up_block1(map2)  #[N,32,h/2,w/2]
        map1=self.up2(map2,map1)  #[N,32,h,w]
        map1=self.up_block2(map1)  #[N,16,h,w]
        out=self.conv2(map1)  #[N,2,h,w]
        
        return out
           
        
        
        
        
        
        
        
        
        
        
        

        
        
       
        
        
        
    
