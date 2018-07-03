import numpy as np
import os
import SimpleITK as sitk
from PerformSegmenation import BreastSeg,HeartSeg,DenseSeg
from PerformMapping import get_median_value,gen_norm,plot_mapping
import argparse
import torch
import time
from Unet_3D import UNet3D
parser = argparse.ArgumentParser(description='Normalization of DCE-MRI')
parser.add_argument('--cuda', type=int, default='1', required=False, help='Run in GPU')
parser.add_argument('--pre', type=str, default='Image_pre.nii.gz', required=True, help='Image path for pre-constrast image')
parser.add_argument('--post1', type=str, default='Image_post1.nii.gz', required=True, help='Image path for post-constrast1 image')
parser.add_argument('--post2', type=str, default='empty', required=False, help='Image path for post-constrast2 image, if any')
parser.add_argument('--post3', type=str, default='empty', required=False, help='Image path for post-constrast3 image, if any')
parser.add_argument('--post4', type=str, default='empty', required=False, help='Image path for post-constrast4 image, if any')
parser.add_argument('--post5', type=str, default='empty', required=False, help='Image path for post-constrast5 image, if any')
parser.add_argument('--post6', type=str, default='empty', required=False, help='Image path for post-constrast6 image, if any')
parser.add_argument('--outfolder',type=str,default='../Results',required=False,help='Folder for saving results')
opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda 0")



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
    
t0 = time.time()
heart = HeartSeg(I_post,opt,model)  
t1 = time.time()  
print 'time for performing heart segmentation - >', t1 - t0
breast = BreastSeg(I_pre,opt,model)
t2 = time.time()
print 'time for performing breast segmentation - >', t2 - t1
dense = DenseSeg(I_pre,breast,opt,model)
t3 = time.time()
print 'time for performing dense tissue segmentation - >', t3 - t2

# extract median values of tissues
airvalue, fatvalue, densevalue, heartvalue, maxvalue = get_median_value(breast \
    ,dense,heart,np.array(sitk.GetArrayFromImage(I_pre)),np.array(sitk.GetArrayFromImage(I_post)))


# perform image normalization
img_pre = gen_norm(I_pre,airvalue, fatvalue, densevalue, heartvalue, maxvalue)
t4 = time.time()
print 'time for performing mapping - >', t4 - t3


print 'total time for normalization - >', time.time() - t0
N_image = sitk.GetImageFromArray(img_pre, isVector=False) 
N_image.SetSpacing(I_pre.GetSpacing())
N_image.SetOrigin(I_pre.GetOrigin())
N_image.SetDirection(I_pre.GetDirection())
sitk.WriteImage(N_image,savepath+'/Norm_pre.nii.gz')    
for i_post in range(1,7):
    postpath = getattr(opt,"post{0}".format(i_post) )
    if postpath !="empty":
        I_post = sitk.ReadImage(postpath)  
        img_post = gen_norm(I_post,airvalue, fatvalue, densevalue, heartvalue, maxvalue)

        
        N_image = sitk.GetImageFromArray(img_post, isVector=False) 
        N_image.SetSpacing(I_post.GetSpacing())
        N_image.SetOrigin(I_post.GetOrigin())
        N_image.SetDirection(I_post.GetDirection())
        sitk.WriteImage(N_image,savepath+'/Norm_post{0}.nii.gz'.format(i_post))    
    
    
# plot mapping
plot_mapping(airvalue, fatvalue, densevalue, heartvalue, maxvalue,savepath)
