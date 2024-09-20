import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import argparse

from networks.DispOccNet import DispNet, OccNet
from Dataset import LFDenseTrainData, LFDenseValData
from Loss import UnDepthOccLoss
from utils.tools import Warp


def adjust_learning_rate(optimizer,lr_init,epoch,lr_step):
    lr = lr_init* (0.8 ** (epoch // lr_step))
    if lr>=1e-4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
        
        
def train(args):
    step=0
    lr_step=50
    epoch_dict,mse_dict={},{}
    for i in range(1,args.save_num+1):
        epoch_dict[str(i)]=0
        mse_dict[str(i)]=0
    best_mse=float('inf')
    best_epoch=0
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    train_set=LFDenseTrainData(args.train_root,args.crop_size)
    val_set=LFDenseValData(args.val_root)
    
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size*2,shuffle=False)
    
    disp_sample=np.linspace(-12,12,num=24)
    res_sample=np.linspace(-1.2,1.2,num=24)
    disp_model=DispNet(in_channels=3,base_channels=24,filter_num=3,disp_sample=disp_sample,res_sample=res_sample,group_num=16)
    print('disp_model parameters: {}'.format(sum(param.numel() for param in disp_model.parameters())))
    occ_model=OccNet(in_channels=7,base_channels=16)
    print('occ_model parameters: {}'.format(sum(param.numel() for param in occ_model.parameters())))
    
    criterion=UnDepthOccLoss()
    
    if torch.cuda.is_available():
        is_cuda=True 
    else:
        is_cuda=False
    if is_cuda:
        disp_model.cuda() 
        occ_model.cuda()
    if args.pre_train:
        checkpoint=torch.load(args.model_path)
        disp_model.load_state_dict(checkpoint['Disp'])
        occ_model.load_state_dict(checkpoint['Occ'])
          
    optimizer = torch.optim.Adam(itertools.chain(disp_model.parameters(),occ_model.parameters()),lr=args.lr_init)
    
    for epoch in range(args.max_epoch):
        disp_model.train()
        occ_model.train()
        if epoch % lr_step==0:
            adjust_learning_rate(optimizer,args.lr_init,epoch,lr_step)
        
        for idx,inp in enumerate(train_loader):
            inp=Variable(inp)
            if is_cuda:
                inp=inp.cuda()
            optimizer.zero_grad()
            #inp: [N,3,3,h,w]
            
            img_ls=inp[:,0,:,:,:] #[N,3,h,w]
            img_ref=inp[:,1,:,:,:] #[N,3,h,w]
            img_rs=inp[:,2,:,:,:] #[N,3,h,w]
        
            disp_init,disp=disp_model(img_ls,img_ref,img_rs)  #[N,1,h,w]
            warp_left_init=Warp(img_ls,disp_init,x_dist=1,y_dist=0)  #[N,3,h,w]
            warp_right_init=Warp(img_rs,disp_init,x_dist=-1,y_dist=0)
            warp_left=Warp(img_ls,disp,x_dist=1,y_dist=0)  #[N,3,h,w]
            warp_right=Warp(img_rs,disp,x_dist=-1,y_dist=0)
            
            occ_map_init=occ_model(warp_left_init,warp_right_init,disp_init)  #[N,2,h,w]
            occ_map=occ_model(warp_left,warp_right,disp)  #[N,2,h,w]
            
            warp_left_cat=torch.cat([warp_left_init,warp_left],dim=0)  #[2N,3,h,w]
            warp_right_cat=torch.cat([warp_right_init,warp_right],dim=0)  #[2N,3,h,w]
            occ_map_cat=torch.cat([occ_map_init,occ_map],dim=0)  #[2N,2,h,w]
            
            ref_syn=warp_left_cat*occ_map_cat[:,0:1,:,:]+warp_right_cat*occ_map_cat[:,1:,:,:]  #[2N,3,h,w]
    
            loss=criterion(disp_init,disp,warp_left_init,warp_right_init,warp_left,warp_right,occ_map_init,occ_map,img_ref,ref_syn)
            loss.backward()
            optimizer.step()
            print('Epoch: %i, batch_idx: %i, train_loss: %f' %(epoch,idx,loss.item()))
            print('')
            step+=1
        
        disp_model.eval()
        with torch.no_grad():
            mse_list=[]
            
            for inp,disp_gt in val_loader:
                if is_cuda:
                    inp,disp_gt=inp.cuda(),disp_gt.cuda()
                #inps: [N,3,3,h,w]
                img_ls=inp[:,0,:,:,:] #[N,3,h,w]
                img_ref=inp[:,1,:,:,:] #[N,3,h,w]
                img_rs=inp[:,2,:,:,:] #[N,3,h,w]
                
                _,disp_pred=disp_model(img_ls,img_ref,img_rs)
                disp_pred=disp_pred/2
                mse=nn.MSELoss()(disp_pred,disp_gt)
                mse_list.append(mse.item())
                        
            ave_mse=np.mean(mse_list)
            print('Epoch {}, average MSE {}'.format(epoch,ave_mse))
            print('')
    
        #save models            
        if epoch<args.save_num:
            torch.save({'Disp':disp_model.state_dict(),'Occ':occ_model.state_dict()},os.path.join(args.save_root,'model%s.pth'%str(epoch+1)))
            mse_dict[str(epoch+1)]=ave_mse
            epoch_dict[str(epoch+1)]=epoch
        else:
            if ave_mse<max(mse_dict.values()):
                torch.save({'Disp':disp_model.state_dict(),'Occ':occ_model.state_dict()},
                            os.path.join(args.save_root,'model%s.pth'%(max(mse_dict,key=lambda x: mse_dict[x]))))
                epoch_dict[max(mse_dict,key=lambda x: mse_dict[x])]=epoch
                mse_dict[max(mse_dict,key=lambda x: mse_dict[x])]=ave_mse
                
        if ave_mse<best_mse:
            best_mse=ave_mse
            best_epoch=epoch
        print('Best MSE {}, epoch {}'.format(best_mse,best_epoch))
        print('Epoch {}'.format(epoch_dict))
        print('MSE {}'.format(mse_dict))
        print('')


if __name__=='__main__':  
    
    parser = argparse.ArgumentParser(description='LF disparity estimation')
    parser.add_argument('--train_root', default='', type=str)
    parser.add_argument('--val_root', default='', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--crop_size',default=256,type=int)
    parser.add_argument('--lr_init', default=1e-3, type=float)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--max_epoch',default=300,type=int)
    parser.add_argument('--save_root',default='',type=str)
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
        