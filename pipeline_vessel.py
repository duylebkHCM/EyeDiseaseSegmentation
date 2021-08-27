import os
import torch
from catalyst import utils
from pytorch_toolbelt.utils.random import set_manual_seed
from datetime import datetime
import argparse
import logging
import subprocess

from src.main.config import BaseConfig, TestConfig
from src.main.train_vessel import train_model
from src.main.tta_vessel import *
from src.main.stat_result_vessel import export_result

logging.basicConfig(level=logging.INFO)

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--createprob', default='false', type=str, help='Just create a probability mask not binary')
    parse.add_argument('--optim_thres', default=0.0, help='Optimal threshold obtain from AUC-PR curve')
    parse.add_argument('--best', default='true', type=str, 
                        help='Using best checkpoint or last checkpoint')
    parse.add_argument('--tta', default='d4', 
                        help='Test Time Augmentation, available:d4, multiscale, flip, hflip, five_crop, ten_crop')
    args = vars(parse.parse_args())

    return args

def start_experiment(args):
    n_devices = torch.cuda.device_count()
    logging.info(f'Start using {n_devices} GPUs')
    exp_name = datetime.now().strftime("%b%d_%H_%M")
    # exp_name = 'Jun08_11_35'
    logging.info(f'Performing experiment {exp_name}')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in range(n_devices)])
    SEED = 1999
    logging.info(f'SEED: {SEED}')
    set_manual_seed(SEED)   
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=False, benchmark=True)


    logging.info("""
    *************************************************************
    *                                                           *
    *                          TRAINING                         *                               
    *                                                           *
    *************************************************************
    """)
    configs = BaseConfig.get_all_attributes()
    train_model(exp_name, configs, SEED)

    logging.info("""
    *************************************************************
    *                                                           *
    *                          INFERENCE                        *                               
    *                                                           *
    *************************************************************
    """)

    configs = TestConfig.get_all_attributes()
    logdir = os.path.join("models/", configs['dataset_name'], configs['lesion_type'], exp_name)
    args['createprob'] = 'true'
    if configs['data_type'] == 'all':
        test_tta(logdir, configs, args)
    else:
        tta_patches(logdir, configs, args)

    # logging.info("""
    # *************************************************************
    # *                                                           *
    # *                          AUC-PR                           *                               
    # *                                                           *
    # *************************************************************
    # """)

    # get_auc(exp_name, configs)
    # optimal_threshold, optimal_threshold_1 = plot_aucpr_curve(exp_name, configs)
    # logging.info(f"Optimal threshold is {optimal_threshold}")

    # logging.info("""
    # *************************************************************
    # *                                                           *
    # *                          BINARY MASK                      *                               
    # *                                                           *
    # *************************************************************
    # """)
    # args['createprob'] = 'false'
    # args['optim_thres'] = round(optimal_threshold, 3)
    # if configs['data_type'] == 'all':
    #     test_tta(logdir, configs, args)
    # else:
    #     tta_patches(logdir, configs, args)
    
    # #@TODO
    # #Visualize predicted vs groundtruth 

    logging.info("""
    *************************************************************
    *                                                           *
    *                          ANALYSIS RESULTS                 *                               
    *                                                           *
    *************************************************************
    """)
    export_result(os.path.join(configs["lesion_type"], exp_name), configs)

    logging.info("""
    *************************************************************
    *                                                           *
    *                          FINISH EXPERIMENT                *                               
    *                                                           *
    *************************************************************
    """)

if __name__ == '__main__':
    subprocess.call(['wget', 'https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage', '-O', '/usr/local/bin/orca'])
    subprocess.call(['chmod', '+x', '/usr/local/bin/orca'])
    subprocess.call(['apt-get', 'install', 'xvfb', 'libgtk2.0-0', 'libgconf-2-4'])
    args = parse_arg()
    start_experiment(args)