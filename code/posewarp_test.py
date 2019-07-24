import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
import cv2
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import scipy.misc

def test(model_name, gpu_id):
    params = param.get_general_params()
    network_dir = params['model_save_dir'] + '/' + model_name
    work_product_dir  = params['project_dir'] + '/' + 'work_product'



    test_feed = data_generation.create_feed(params, params['data_dir'], 'test')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('posewarp-cvpr2018/data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4),loss=[networks.vgg_loss(vgg_model, response_weights, 12)])
    model.load_weights(network_dir + '/' +'weights_model_gan_improved.h5')


    
    n_iters = params['n_training_iter']
    x, y = next(test_feed)
    
#     src_pose = x[1][0][:,:,0] 
#     trgt_pose = x[2][0][:,:,0]
    src_limb_masks = np.amax(np.asarray(x[3][0]), axis=2)
    src_limb_mask_1 = np.asarray(x[3][0][:,:,0])
    src_limb_mask_2 = np.asarray(x[3][0][:,:,1])
    src_limb_mask_3 = np.asarray(x[3][0][:,:,2])
    src_limb_mask_4 = np.asarray(x[3][0][:,:,3])
    src_limb_mask_5 = np.asarray(x[3][0][:,:,4])
    src_pose = np.amax(np.asarray(x[1][0]), axis=2)
    trgt_pose = np.amax(np.asarray(x[2][0]), axis=2)
#     for i in range(1,7):
#         src_pose = src_pose+ x[1][0][:,:,i]
#         trgt_pose = trgt_pose+ x[2][0][:,:,i]
        
    scipy.misc.imsave(work_product_dir+'/'+'source_pose.jpg',src_pose)
    scipy.misc.imsave(work_product_dir+'/'+'target_pose.jpg',trgt_pose)
    scipy.misc.imsave(work_product_dir+'/'+'source_limb_mask.jpg',src_limb_masks)
#     scipy.misc.imsave(network_dir+'/'+'source_limb_mask_1.jpg',src_limb_mask_1)
#     scipy.misc.imsave(network_dir+'/'+'source_limb_mask_2.jpg',src_limb_mask_2)
#     scipy.misc.imsave(network_dir+'/'+'source_limb_mask_3.jpg',src_limb_mask_3)
#     scipy.misc.imsave(network_dir+'/'+'source_limb_mask_4.jpg',src_limb_mask_4)
#     scipy.misc.imsave(network_dir+'/'+'source_limb_mask_5.jpg',src_limb_mask_5)
#     scipy.misc.imsave(network_dir+'/'+'source_pose_new.jpg',src_pose_n)
    
    target_img = np.asarray(y[0])
    src_img=np.asarray(x[0][0])
    
    yimg = model.predict(x,1)
    gen_img = np.asarray(yimg[0])
    
    constarr = 255*0.5*np.ones((256,256,3))
    
#     scipy.misc.imsave(network_dir+'/'+'source_image.jpg',src_img)
#     scipy.misc.imsave(network_dir+'/'+'target_image.jpg',target_img)
#     scipy.misc.imsave(network_dir+'/'+'generated_target_image.jpg',gen_img)
    
    cv2.imwrite(work_product_dir+'/'+'target.jpg',constarr+np.multiply(target_img,0.5*255)) #target_img)
    cv2.imwrite(work_product_dir+'/'+'gen_target.jpg',constarr+np.multiply(gen_img,0.5*255))
    cv2.imwrite(work_product_dir+'/'+'source.jpg',constarr+np.multiply(src_img,0.5*255)) #src_img)

#     sp = x[1][0]
#     tp = x[2][0]


    
        
#     print("minimum of tpose: ",np.amin(trgt_pose))
#     print("maximum of tpose: ",np.amax(trgt_pose))
# #     poses1 = x[1][0][:,:,0]
#     print(poses1.shape)
#     src_pose = int(src_pose*255)
#     trgt_pose = int(trgt_pose*255)
#     src_pose = np.reshape(src_pose,(np.shape(src_pose)[0],np.shape(src_pose)[1],1))

#     print(src_pose.shape)
#     print(trgt_pose.shape)



#     cv2.imwrite(network_dir+'/'+'source_pose.jpg',np.multiply(src_pose,255))
#     cv2.imwrite(network_dir+'/'+'target_pose.jpg',np.multiply(trgt_pose,255))
#     scipy.misc.imsave(network_dir+'/'+'target_pose.jpg',x[2][0])
#     print(ylabel)
   


if __name__ == "__main__":
    if len(sys.argv) != 3: 
        print("Need model name and gpu id as command line arguments.")
    else:        
        test(sys.argv[1], sys.argv[2])

