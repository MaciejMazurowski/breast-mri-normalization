import numpy as np
import os
import SimpleITK as sitk
from PerformSegmenation import BreastSeg,HeartSeg,DenseSeg
from PerformMapping import get_median_value,gen_norm
from Unet_3D import UNet3D
import torch
def ImageNorm(opt):
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    
    
    savepath = opt.outfolder
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
             
    I_pre = sitk.ReadImage(opt.pre)  
    I_post = sitk.ReadImage(opt.post1)  
    
    
    
    
    # perform image segmentation
    if opt.cuda:
        model = UNet3D(1,1).cuda()
    else:
        model = UNet3D(1,1)

    heart = HeartSeg(I_post,opt,model)  

    breast = BreastSeg(I_pre,opt,model)


    dense = DenseSeg(I_pre,breast,opt,model)


    # extract median values of tissues
    airvalue, fatvalue, densevalue, heartvalue, maxvalue = get_median_value(breast \
        ,dense,heart,np.array(sitk.GetArrayFromImage(I_pre)),np.array(sitk.GetArrayFromImage(I_post)))
    
    
    # perform image normalization
    img_pre = gen_norm(I_pre,airvalue, fatvalue, densevalue, heartvalue, maxvalue)
    

    Heat_image = sitk.GetImageFromArray(img_pre, isVector=False) 
    Heat_image.SetSpacing(I_pre.GetSpacing())
    Heat_image.SetOrigin(I_pre.GetOrigin())
    Heat_image.SetDirection(I_pre.GetDirection())
    sitk.WriteImage(Heat_image,savepath+'/Norm_pre.nii.gz')    
    for i_post in range(1,7):
        postpath = getattr(opt,"post{0}".format(i_post) )
        if postpath !="empty":
            I_post = sitk.ReadImage(postpath)  
            img_post = gen_norm(I_post,airvalue, fatvalue, densevalue, heartvalue, maxvalue)
    
            
            Heat_image = sitk.GetImageFromArray(img_post, isVector=False) 
            Heat_image.SetSpacing(I_post.GetSpacing())
            Heat_image.SetOrigin(I_post.GetOrigin())
            Heat_image.SetDirection(I_post.GetDirection())
            sitk.WriteImage(Heat_image,savepath+'/Norm_post{0}.nii.gz'.format(i_post))    
        
    # plot mapping
   # plot_mapping(airvalue, fatvalue, densevalue, heartvalue, maxvalue,savepath)
