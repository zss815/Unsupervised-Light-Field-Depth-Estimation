import torch
from torch import nn
from utils.losses import SSIM, Smooth_loss


class UnDepthOccLoss(nn.Module):
    def __init__(self):
        super(UnDepthOccLoss,self).__init__()
        self.l1_loss=nn.L1Loss()
        self.ssim=SSIM(window_size=11,size_average=True)

    def forward(self,disp_init,disp,warp_left_init,warp_right_init,warp_left,warp_right,occ_map_init,occ_map,img_ref,ref_syn):
        #Initial disparity  losses
        occ_init1=occ_map_init[:,0:1,:,:]  #[N,1,h,w]
        occ_init2=occ_map_init[:,1:,:,:]   #[N,1,h,w]
        
        #Photometric loss with occlusion
        pm_init_loss=torch.mean(occ_init1*torch.abs(warp_left_init-img_ref))+\
            torch.mean(occ_init2*torch.abs(warp_right_init-img_ref))
           
        ssim_init_loss=(1-self.ssim(warp_left_init,img_ref)+1-self.ssim(warp_right_init,img_ref))/2
        
        smooth_init_loss=Smooth_loss(disp_init,img_ref,gamma=100)
        
        #Final disparity losses
        occ1=occ_map[:,0:1,:,:]  #[N,1,h,w]
        occ2=occ_map[:,1:,:,:]   #[N,1,h,w]
        pm_loss=torch.mean(occ1*torch.abs(warp_left-img_ref))+\
            torch.mean(occ2*torch.abs(warp_right-img_ref))
            
        smooth_loss=Smooth_loss(disp,img_ref,gamma=100)
                
        ssim_loss=(1-self.ssim(warp_left,img_ref)+1-self.ssim(warp_right,img_ref))/2

        #Occlusion losses
        img_ref_cat=torch.cat([img_ref,img_ref],dim=0) #[2N,3,h,w]
        recon_loss=self.l1_loss(ref_syn,img_ref_cat)
        
        occ_map_cat=torch.cat([occ_map_init,occ_map],dim=0)  #[2N,2,h,w]
        
        occ_smooth_loss=Smooth_loss(occ_map_cat[:,0:1,:,:],img_ref_cat,gamma=100)
        
        disp_init_loss=1*pm_init_loss+1*ssim_init_loss+0.1*smooth_init_loss
        disp_loss=1*pm_loss+1*ssim_loss+0.1*smooth_loss
        occ_loss=1*recon_loss+0.01*occ_smooth_loss
        
        total_loss=disp_init_loss+disp_loss+occ_loss
        
        return total_loss
    
    

  
