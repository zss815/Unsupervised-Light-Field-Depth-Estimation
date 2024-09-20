import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import random


def MacroPixel2SAIs(x,ah,aw):
    sai_all=[]
    for i in range(ah):
        for j in range(aw):
            img=x[i::ah, j::aw]
            sai_all.append(img)
    sai_all=np.stack(sai_all)
    return sai_all


def Rotate(imgs):
    N,c,h,w=imgs.size()
    imgs_new=torch.FloatTensor(N,c,h,w).zero_()
    for i in range(N):
        img=imgs[i]
        img=torch.rot90(img,k=1,dims=[1,2])
        imgs_new[i]=img
    
    return imgs_new    

    
class LFDenseTrainData(Dataset):
    def __init__(self,data_root,crop_size):
        super(LFDenseTrainData,self).__init__()
        self.img_path=[]
        for file in os.listdir(data_root):
            if not file.startswith('.'):
                self.img_path.append(os.path.join(data_root,file))
        self.crop_size=crop_size
    
    def __getitem__(self,index):
        mp=np.array(Image.open(self.img_path[index]))
        #normalize
        mp=(mp-np.min(mp))/(np.max(mp)-np.min(mp))      
        sais=MacroPixel2SAIs(mp,ah=9,aw=9)
        
        #crop
        h,w=sais.shape[1],sais.shape[2]
        if h>self.crop_size:
            h_begin=np.random.randint(h-self.crop_size)
        else:
            h_begin=0
        if w>self.crop_size:
            w_begin=np.random.randint(w-self.crop_size)
        else:
            w_begin=0
        sais=sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
        
        idx_h=[[39,40,41],[38,40,42],[37,40,43]]
        idx_v=[[31,40,49],[22,40,58],[13,40,67]]
        
        if np.random.rand()>0.5:
            idx=random.choice(idx_h)
            inp=sais[idx]  #[3,h,w,3]
            inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
            inp=torch.from_numpy(inp).float()
        else:
            idx=random.choice(idx_v)
            inp=sais[idx]
            inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
            inp=torch.from_numpy(inp).float()
            inp=Rotate(inp)  
        
        return inp
    
    def __len__(self):
        return len(self.img_path)
    
    

class LFDenseValData(Dataset):
    def __init__(self,data_root):
        super(LFDenseValData,self).__init__()
        self.img_path=[]
        self.disp_path=[]
        for file in os.listdir(os.path.join(data_root,'Disp')):
            if not file.startswith('.'):
                self.img_path.append(os.path.join(data_root,'MP',file.split('.')[0]+'.png'))
                self.disp_path.append(os.path.join(data_root,'Disp',file))
    
    def __getitem__(self,index):
        mp=np.array(Image.open(self.img_paths[index]))
        #normalize
        mp=(mp-np.min(mp))/(np.max(mp)-np.min(mp))      
        sais=MacroPixel2SAIs(mp,ah=9,aw=9)
            
        idx=[38,40,42]
        inp=sais[idx]    #[3,h,w,3]
        inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
        inp=torch.from_numpy(inp).float()
        
        disp=np.load(self.disp_paths[index])
        disp=torch.from_numpy(disp).float()  #[h,w]
        disp=torch.unsqueeze(disp,dim=0)  #[1,h,w]
        
        return inp,disp
    
    def __len__(self):
        return len(self.img_paths)
    
    
class LFSparseTrainData(Dataset):
    def __init__(self,data_root,crop_size):
        super(LFSparseTrainData,self).__init__()
        self.img_path=[]
        for file in os.listdir(data_root):
            if not file.startswith('.'):
                self.img_path.append(os.path.join(data_root,file))
        self.crop_size=crop_size             
    
    def __getitem__(self,index):
        mp=np.array(Image.open(self.img_path[index]))
        #normalize
        mp=(mp-np.min(mp))/(np.max(mp)-np.min(mp))      
        sais=MacroPixel2SAIs(mp,ah=9,aw=9)
        
        #crop
        h,w=sais.shape[1],sais.shape[2]
        if h>self.crop_size:
            h_begin=np.random.randint(h-self.crop_size)
        else:
            h_begin=0
        if w>self.crop_size:
            w_begin=np.random.randint(w-self.crop_size)
        else:
            w_begin=0
        sais=sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
        
        idx_h=[39,40,41]
        idx_v=[31,40,49]
        
        if np.random.rand()>0.5: 
            inp=sais[idx_h]    #[3,h,w,3]
            inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
            inp=torch.from_numpy(inp).float()
        else:
            inp=sais[idx_v]
            inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
            inp=torch.from_numpy(inp).float()
            inp=Rotate(inp)

        return inp
    
    def __len__(self):
        return len(self.img_path)
 
    
class LFSparseValData(Dataset):
    def __init__(self,data_root):
        super(LFSparseValData,self).__init__()
        self.img_path=[]
        self.disp_path=[]
        for file in os.listdir(os.path.join(data_root,'Disp')):
            if not file.startswith('.'):
                self.img_path.append(os.path.join(data_root,'MP',file.split('.')[0]+'.png'))
                self.disp_path.append(os.path.join(data_root,'Disp',file))
    
    def __getitem__(self,index):
        mp=np.array(Image.open(self.img_path[index]))
        #normalize
        mp=(mp-np.min(mp))/(np.max(mp)-np.min(mp))      
        sais=MacroPixel2SAIs(mp,ah=9,aw=9)
            
        idx=[39,40,41]
        inp=sais[idx]    #[3,h,w,3]
        inp=np.transpose(inp,axes=(0,3,1,2))   #[3,3,h,w]
        inp=torch.from_numpy(inp).float()
        
        disp=np.load(self.disp_path[index]) 
        disp=torch.from_numpy(disp).float()  #[h,w]
        disp=torch.unsqueeze(disp,dim=0)  #[1,h,w]
        
        return inp,disp
    
    def __len__(self):
        return len(self.img_path)
    
    
    

