"""
Various important parameters of our model and training procedure.
"""


def get_general_params():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = int(256/dn)
    param['IMG_WIDTH'] = int(256/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 7
    param['n_limbs'] = 5

    # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    
    #0:'Head'	1:'Right wrist'	2:'Left wrist'	3:'Right elbow'	4:'Left elbow'	5:'Right shoulder'	6:'Left shoulder'
    
    param['limbs'] = [ [1, 3], [3, 5], [2, 4], [4, 6], [0, 5, 6]]

    param['n_training_iter'] = 1000 #200000
    param['test_interval'] = 10
    param['model_save_interval'] = 20
    param['project_dir'] = 'posewarp-cvpr2018'
    param['model_save_dir'] = param['project_dir'] + '/models'
    param['data_dir'] = 'posewarp-cvpr2018/data'
    param['batch_size'] = 4
    return param

