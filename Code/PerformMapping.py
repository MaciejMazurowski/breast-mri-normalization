import numpy as np
import scipy.io as sio
import SimpleITK as sitk
from scipy import ndimage as nd  
import warnings

def get_median_value(breast,dense,heart,preimage,postimage):
    struct = nd.generate_binary_structure(3, 3)
    breast = nd.morphology.binary_erosion(breast,structure=struct,iterations=8)

    heart_index = np.nonzero(heart)
    fat_index = np.nonzero(np.logical_and(breast==1, dense==0))
    dense_index = np.nonzero(dense)
    
    I_h = np.sort(postimage[heart_index[0],heart_index[1],heart_index[2]].flatten())
    I_f = np.sort(preimage[fat_index[0],fat_index[1],fat_index[2]].flatten())
    I_d = np.sort(preimage[dense_index[0],dense_index[1],dense_index[2]].flatten())
    heartvalue = I_h[-int(0.10*len(I_h))]
    fatvalue= I_f[len(I_f)/2]
    densevalue = I_d[len(I_d)/2]
    
    I_sort = np.sort(postimage.flatten())
    maxvalue=   I_sort[-int(0.001*len(I_sort))]
    airvalue=   I_sort[int(0.01*len(I_sort))]

    if heartvalue<densevalue:
        densevalue= heartvalue*987/1756
        warnings.warn("the value of dense tissue should be smaller than the value of heart")
    if fatvalue>densevalue:
        fatvalue= I_f[int(0.2*len(I_f))]
        densevalue = I_d[int(0.8*len(I_d))]

    if fatvalue>densevalue:
        fatvalue= I_f[int(0.1*len(I_f))]
        densevalue = I_d[int(0.9*len(I_d))]       
    if fatvalue>densevalue:
        fatvalue = 287/987 * densevalue
        warnings.warn("the value of fat should be smaller than the value of dense tissue")

    return airvalue, fatvalue, densevalue, heartvalue, maxvalue
        

def get_template_spacevalue():
    intensitys= sio.loadmat('IntensityDistrabution_tissues_train_pre.mat')
    densevalue = np.array(intensitys['densevalue'])
    fatvalue = np.array(intensitys['breastvalue'])
    airvalue = np.array(intensitys['airvalue'])
    intensitys= sio.loadmat('IntensityDistrabution_tissues_train_post.mat')
    heartvalue = np.array(intensitys['heartvalue'])
    maxvalue = np.array(intensitys['maxvalue'])

    ht = np.ceil(np.median(heartvalue))
    dt = np.ceil(np.median(densevalue))
    ft = np.ceil(np.median(fatvalue))
    at = np.ceil(np.median(airvalue))
    mt = np.ceil(np.median(maxvalue))
    
    in_heartvalue = heartvalue-ht
    in_densevalue = densevalue-dt
    in_fatvalue = fatvalue-ft
    in_maxvalue = maxvalue-mt
    
    index_all = (np.abs(in_heartvalue)+np.abs(in_densevalue)+np.abs(in_fatvalue)+np.abs(in_maxvalue))/4
    
    
    index_im = np.where(index_all==np.min(index_all))

    ht = heartvalue[index_im[0],index_im[1]]
    dt = densevalue[index_im[0],index_im[1]]
    ft = fatvalue[index_im[0],index_im[1]]
    at = airvalue[index_im[0],index_im[1]]
    mt = maxvalue[index_im[0],index_im[1]]
    return at, dt, ht, mt,ft

import matplotlib.pyplot as plt
def plot_mapping(airvalue, fatvalue, densevalue, heartvalue, maxvalue,savepath):
    at, dt, ht, mt,ft = get_template_spacevalue()
               
    mt_ = (maxvalue - heartvalue) * (ht-dt)/(heartvalue-densevalue)+ ht
    y = [at,ft,dt,ht,mt_]
    
    x = [airvalue,fatvalue,densevalue,heartvalue,maxvalue]

    transp =1
    plt.figure()
    plt.scatter(x[0],y[0],c='r',alpha=transp,label='Air')
    plt.scatter(x[1],y[1],c='g',alpha=transp,label='Fat Tissue')
    plt.scatter(x[2],y[2],c='b',alpha=transp,label='Dense Tissue')
    plt.scatter(x[3],y[3],c='y',alpha=transp,label='Heart ')
    plt.plot(x,y,'-')


    plt.xlabel('Subject space')
    plt.ylabel('Normalized Space')
    plt.legend(loc='upper left')
#    plt.show()
    plt.savefig(savepath+'/mapping.png')


def gen_norm(I,airvalue, fatvalue, densevalue, heartvalue, maxvalue):
    at, dt, ht, mt,ft = get_template_spacevalue()

    image = np.array(sitk.GetArrayFromImage(I))   
    image2= image*1
    
    

    image2[np.logical_and(image>=airvalue,image<fatvalue)] = \
           (image[np.logical_and(image>=airvalue,image<fatvalue)] -airvalue) \
           *(ft-at)/(fatvalue-airvalue)+at
    
    image2[np.logical_and(image>=fatvalue,image<densevalue)] = \
           (image[np.logical_and(image>=fatvalue,image<densevalue)] -fatvalue) \
           *(dt-ft)/(densevalue-fatvalue)+ft
           
           
    image2[np.logical_and(image>=densevalue,image<heartvalue)] = \
           (image[np.logical_and(image>=densevalue,image<heartvalue)] - densevalue )\
                 *(ht-dt)/(heartvalue-densevalue)  +dt
           
    image2[image>=heartvalue] = \
           (image[image>=heartvalue] - heartvalue )\
                 *(ht-dt)/(heartvalue-densevalue)+ ht  
    image2 = image2.astype('uint16')              
    return image2


          

