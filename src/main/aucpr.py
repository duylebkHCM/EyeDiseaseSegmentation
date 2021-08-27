from PIL import Image
import numpy as np
import os
import re
import sys
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_auc_score
from pathlib import Path
from tqdm .auto import tqdm
import plotly.express as px
import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)

from .util import lesion_dict

def get_auc(generator: Callable, config):    
    sum_pav = 0
    i =  0

    for pred_mask, gt_mask, _ in generator:
        if gt_mask.sum() == 0:
            continue
        pav = average_precision_score(gt_mask.reshape(-1), pred_mask.reshape(-1))
        print('PAV', pav)
        sum_pav += pav
        i += 1

    mpav = sum_pav / i
    return mpav

def get_aucroc(generator: Callable, config):    
    sum_pav = 0
    i =  0
    for pred_mask, gt_mask, _ in generator:
        if gt_mask.sum() == 0:
            continue
        pav = roc_auc_score(gt_mask.reshape(-1), pred_mask.reshape(-1))
        sum_pav += pav
        i += 1

    mpav = sum_pav / i
    return mpav

def plot_aucpr_curve(generator: Callable, exp_name, test_config):
    # gt_dir = test_config['test_mask_path'] / lesion_dict[test_config['lesion_type']].dir_name
    # prob_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'] ,'tta', test_config['lesion_type'], 'prob_image', exp_name) 
    figure_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'], 'figures', test_config['lesion_type']) 

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    thresh_list = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    thresh_size = len(thresh_list)
    sn = np.empty(thresh_size, dtype=float)
    ppv = np.empty(thresh_size, dtype=float)
    thresh_array = np.array(thresh_list)

    true_p, actual_p, pred_p = {}, {}, {}
    for pred_mask, gt_mask, _ in generator:
        for th in range(thresh_size):
            threshold = thresh_array[th]   
            arr_pred = (pred_mask > threshold).astype('uint8')
            tp = np.sum(gt_mask & arr_pred)
            ap = np.sum(gt_mask)
            pp = np.sum(arr_pred)

            if true_p.get(str(threshold), None) is None:
                true_p[str(threshold)] = tp
            else:
                true_p[str(threshold)] += tp    

            if actual_p.get(str(threshold), None) is None:
                actual_p[str(threshold)] = ap
            else:
                actual_p[str(threshold)] += ap    

            if pred_p.get(str(threshold), None) is None:
                pred_p[str(threshold)] = pp
            else:
                pred_p[str(threshold)] += pp        

    for th in range(thresh_size):
        threshold = thresh_array[th]
        sn[th] = (float(true_p[str(threshold)]) + 1e-7)/(float(actual_p[str(threshold)])+ 1e-7)
        ppv[th] = (float(true_p[str(threshold)]) +  1e-7)/(float(pred_p[str(threshold)]) + 1e-7)

    recall = np.array(sn)
    precision = np.array(ppv)
    f_score = (2*recall*precision) / (recall + precision)
    aucpr = auc(recall, precision)
    #https://www.kaggle.com/nicholasgah/optimal-probability-thresholds-using-pr-curve
    optimal_threshold = sorted(list(zip(
        np.abs(precision - recall), thresh_list)), key=lambda i: i[0], reverse=False)[0][1]

    optimal_threshold_1 = sorted(list(zip(np.sqrt((1-precision)**2 + (1-recall)**2), thresh_list)), key=lambda i: i[0], reverse=False)[0][1]

    optimal_threshold_2 = sorted(list(zip(f_score, thresh_list)), key=lambda i: i[0], reverse=True)[0][1]
    logging.info(f'OPTIMAL THRESHOLD: {optimal_threshold}')
    logging.info(f'OPTIMAL THRESHOLD 1: {optimal_threshold_1}')
    logging.info(f'OPTIMAL THRESHOLD 2: {optimal_threshold_2}')

    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve AUC:{aucpr}-Optimal threshold: {optimal_threshold_2}',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_image(figure_dir + "/{}.jpg".format(str(exp_name)))
    logging.info(f'Saved AUC-PR Curve to {figure_dir}')

    return optimal_threshold, optimal_threshold_1, optimal_threshold_2

def plot_aucroc_curve(generator: Callable, exp_name, test_config):
    # gt_dir = test_config['test_mask_path'] / lesion_dict[test_config['lesion_type']].dir_name
    # prob_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'] ,'tta', test_config['lesion_type'], 'prob_image', exp_name) 
    figure_dir = os.path.join(test_config['out_dir'], test_config['dataset_name'], 'figures') 

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    thresh_list = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    thresh_size = len(thresh_list)
    sn = np.empty(thresh_size, dtype=float)
    sp = np.empty(thresh_size, dtype=float)
    ppv = np.empty(thresh_size, dtype=float)
    thresh_array = np.array(thresh_list)

    true_p, actual_p, pred_p, true_n, actual_n = {}, {}, {}, {}, {}
    for pred_mask, gt_mask, _ in generator:
        for th in range(thresh_size):
            threshold = thresh_array[th]   
            arr_pred = (pred_mask > threshold).astype('uint8')
            tp = np.sum(gt_mask & arr_pred)
            ap = np.sum(gt_mask)
            pp = np.sum(arr_pred)
            an = gt_mask.shape[0]*gt_mask.shape[1] - ap
            fp = pp - tp
            tn = an - fp

            if true_p.get(str(threshold), None) is None:
                true_p[str(threshold)] = tp
            else:
                true_p[str(threshold)] += tp    

            if actual_p.get(str(threshold), None) is None:
                actual_p[str(threshold)] = ap
            else:
                actual_p[str(threshold)] += ap    

            if pred_p.get(str(threshold), None) is None:
                pred_p[str(threshold)] = pp
            else:
                pred_p[str(threshold)] += pp      

            if true_n.get(str(threshold), None) is None:
                true_n[str(threshold)] = tn
            else:
                true_n[str(threshold)] += tn  

            if actual_n.get(str(threshold), None) is None:
                actual_n[str(threshold)] = an
            else:
                actual_n[str(threshold)] += an  


    for th in range(thresh_size):
        threshold = thresh_array[th]
        sn[th] = (float(true_p[str(threshold)]) + 1e-7)/(float(actual_p[str(threshold)])+ 1e-7)
        sp[th] = (float(true_n[str(threshold)]) +  1e-7)/(float(actual_n[str(threshold)]) + 1e-7)
        ppv[th] = (float(true_p[str(threshold)]) +  1e-7)/(float(pred_p[str(threshold)]) + 1e-7)

    tpr = np.array(sn)
    fpr = 1 - np.array(sp)
    precision = np.array(ppv)
    aucroc = auc(fpr, tpr)
    #https://www.kaggle.com/nicholasgah/optimal-probability-thresholds-using-pr-curve
    f_score = (2*tpr*precision) / (tpr + precision)
    idx = np.argmax(f_score)
    optimal_threshold= thresh_list[idx]

    logging.info(f'OPTIMAL THRESHOLD: {optimal_threshold}')

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve AUC:{aucroc}-Optimal threshold: {optimal_threshold}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_image(figure_dir + "/{}.jpg".format(str(exp_name)))
    logging.info(f'Saved AUC-ROC Curve to {figure_dir}')

    return optimal_threshold
