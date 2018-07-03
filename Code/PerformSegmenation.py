import numpy as np
import torch
from scipy import ndimage as nd  
import torch.nn.parallel
import SimpleITK as sitk
from torch.autograd import Variable

from Segmenation import Generator_multichannels,Chunks_Image

#def imgnorm(N_I,index1=0.05,index2=0.05):
#    I_sort = np.sort(N_I.flatten())
#    I_min = I_sort[int(index1*len(I_sort))]
#    I_max = I_sort[-int(index2*len(I_sort))]
#    
#    N_I =255.0*(N_I-I_min)/(I_max-I_min)
#    N_I[N_I>255.0]=255.0
#    N_I[N_I<0.0]=0.0
#    N_I2 = N_I.astype(np.float32)
#    
#    return N_I2


def BreastSeg(I,opt,model):
    modelpath = "../Model/"       
    modelname = modelpath+"/model_breast.pth"
    checkpoint = torch.load(modelname)
    model.load_state_dict(checkpoint)


    numofseg =1

    commonspacing = [3,3,3]

    image=np.array(sitk.GetArrayFromImage(I))

    imageshape = image.shape

    scale_subject =np.array(I.GetSpacing())
    scale_subject = scale_subject[::-1]/commonspacing    
    image = nd.interpolation.zoom(image,scale_subject,order=1)    
    imagesize = image.shape
    image= (image-np.mean(image))/np.std(image)    
    
    sizeofchunk = 16
    sizeofchunk_expand = 48    
    if opt.cuda:
        sizeofchunk = 72
        sizeofchunk_expand = 160
    image_one = np.zeros((1,imagesize[0],imagesize[1],imagesize[2]),dtype='float32')
    image_one[0,...] =image 
    chunk_batch, nb_chunks, idx_xyz, sizeofimage = Generator_multichannels(image_one,sizeofchunk,sizeofchunk_expand,1)
 
    seg_batch = chunk_batch*0.0

    for i_chunk in range(np.size(chunk_batch,0)):
        input = Variable(torch.from_numpy(chunk_batch[i_chunk:i_chunk+1,...]),volatile=True)
        model.eval()
        if opt.cuda:
            input = input.cuda()
        prediction = model(input)   
        
        seg_batch[i_chunk,0,...] = (prediction.data).cpu().numpy()
        
    
    for i_seg in range(numofseg):
        prob_image = Chunks_Image(seg_batch[:,i_seg:i_seg+1,...], nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage)
        up_image = nd.interpolation.zoom(prob_image,1/scale_subject,order=1)
        up_image_norm = np.zeros(imageshape,dtype='float32')
        temp_image = up_image[0:imageshape[0],0:imageshape[1],0:imageshape[2]]
        shape_tempimage =np.shape(temp_image)
        up_image_norm[0:shape_tempimage[0],0:shape_tempimage[1],0:shape_tempimage[2]] = temp_image
        
        threshold = 0.5
        idx = up_image_norm > threshold
        up_image_norm[idx] = 1
        up_image_norm[~idx] = 0
        struct = nd.generate_binary_structure(3, 3)
        up_image_norm = nd.morphology.binary_dilation(up_image_norm,structure=struct,iterations=3)
        up_image_norm = nd.morphology.binary_erosion(up_image_norm,structure=struct,iterations=3)                
    
    return up_image_norm

def HeartSeg(I,opt,model):
    modelpath = "../Model/"
    modelname = modelpath+"/model_heart.pth"
    checkpoint = torch.load(modelname)
    model.load_state_dict(checkpoint)


    numofseg =1

    commonspacing = [3,3,3]
    

    image=np.array(sitk.GetArrayFromImage(I))

    imageshape = image.shape

    scale_subject =np.array(I.GetSpacing())
    scale_subject = scale_subject[::-1]/commonspacing    
    image = nd.interpolation.zoom(image,scale_subject,order=1)    
    imagesize = image.shape
    image= (image-np.mean(image))/np.std(image)     
    sizeofchunk = 16
    sizeofchunk_expand = 48     
    if opt.cuda:
        sizeofchunk = 72
        sizeofchunk_expand = 160
    image_one = np.zeros((1,imagesize[0],imagesize[1],imagesize[2]),dtype='float32')
    image_one[0,...] =image 
    chunk_batch, nb_chunks, idx_xyz, sizeofimage = Generator_multichannels(image_one,sizeofchunk,sizeofchunk_expand,1)
 
    seg_batch = chunk_batch*0.0

    for i_chunk in range(np.size(chunk_batch,0)):
        input = Variable(torch.from_numpy(chunk_batch[i_chunk:i_chunk+1,...]),volatile=True)
        model.eval()
        if opt.cuda:
            input = input.cuda()
        prediction = model(input)   
        
        seg_batch[i_chunk,0,...] = (prediction.data).cpu().numpy()
        
    
    for i_seg in range(numofseg):
        prob_image = Chunks_Image(seg_batch[:,i_seg:i_seg+1,...], nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage)
        up_image = nd.interpolation.zoom(prob_image,1/scale_subject,order=1)
        up_image_norm = np.zeros(imageshape,dtype='float32')
        temp_image = up_image[0:imageshape[0],0:imageshape[1],0:imageshape[2]]
        shape_tempimage =np.shape(temp_image)
        up_image_norm[0:shape_tempimage[0],0:shape_tempimage[1],0:shape_tempimage[2]] = temp_image
        
        threshold = 0.5
        idx = up_image_norm > threshold
        up_image_norm[idx] = 1
        up_image_norm[~idx] = 0
            
   
    return up_image_norm

def DenseSeg(I,maskimg,opt,model):
    modelpath = "../Model/"

    modelname = modelpath+"/model_densetissue.pth"
    checkpoint = torch.load(modelname)
    model.load_state_dict(checkpoint)

    numofseg = 1
    numofchannels = 1
    image=np.array(sitk.GetArrayFromImage(I))
    commonspacing = [2,2,2]
    
    struct = nd.generate_binary_structure(3, 3)
    
    maskimg = nd.morphology.binary_erosion(maskimg,structure=struct,iterations=3)

    image=image*maskimg
    imageshape = image.shape
    scale_subject =np.array(I.GetSpacing())
    scale_subject = scale_subject[::-1]/commonspacing    
    image = nd.interpolation.zoom(image,scale_subject,order=1)    
    imagesize = image.shape
    image= (image-np.mean(image))/np.std(image)     
    sizeofchunk = 16
    sizeofchunk_expand = 48    
    if opt.cuda:
        sizeofchunk = 72
        sizeofchunk_expand = 160
    image_one = np.zeros((numofchannels,imagesize[0],imagesize[1],imagesize[2]),dtype='float32')
    image_one[0,...] =image 
    chunk_batch, nb_chunks, idx_xyz, sizeofimage = Generator_multichannels(image_one,sizeofchunk,sizeofchunk_expand,numofchannels)
 
    seg_batch = chunk_batch*0.0

    for i_chunk in range(np.size(chunk_batch,0)):
        input = Variable(torch.from_numpy(chunk_batch[i_chunk:i_chunk+1,...]),volatile=True)
        model.eval()
        if opt.cuda:
            input = input.cuda()
        prediction = model(input)   
        
        seg_batch[i_chunk,0,...] = (prediction.data).cpu().numpy()
        
    
    for i_seg in range(numofseg):
        prob_image = Chunks_Image(seg_batch[:,i_seg:i_seg+1,...], nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage)
        up_image = nd.interpolation.zoom(prob_image,1/scale_subject,order=1)
        up_image_norm = np.zeros(imageshape,dtype='float32')
        temp_image = up_image[0:imageshape[0],0:imageshape[1],0:imageshape[2]]
        shape_tempimage =np.shape(temp_image)
        up_image_norm[0:shape_tempimage[0],0:shape_tempimage[1],0:shape_tempimage[2]] = temp_image
        
        threshold = 0.999
        idx = up_image_norm > threshold
        up_image_norm[idx] = 1
        up_image_norm[~idx] = 0
                 

    return up_image_norm
