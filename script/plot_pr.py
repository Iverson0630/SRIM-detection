# =========================================================
# @purpose: plot PR curve by COCO API and mmdet API
# @date：   2020/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/mmdetection_plot_pr_curve
# =========================================================

import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset

MODEL = "faster_rcnn"
MODEL_NAME = "cascade_rcnn_r50_fpn_1x_coco"

CONFIG_FILE = f"configs/zst/faster_rcnn_r50_caffe_mstrain_1x_coco_no_fpn.py"
RESULT_FILE_OUR = f"eval/pr/pkl/our.pkl"
RESULT_FILE_MIX = f"eval/pr/pkl/mixing.pkl"
RESULT_FILE_SF = f"eval/pr/pkl/finetune.pkl"
RESULT_FILE_TR = f"eval/pr/pkl/target_only.pkl"
RESULT_FILE_AD = f"eval/pr/pkl/advs.pkl"

def get_precise(config_file, result_file ,metric ):
    cfg = Config.fromfile(config_file)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

   # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric]) 

    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    
    return precisions
    
def plot_pr_curve(precisions, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """
    
 
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3 
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array1 = precisions[0, :, 0, 0, 2] 
    pr_array2 = precisions[1, :, 0, 0, 2] 
    pr_array3 = precisions[2, :, 0, 0, 2] 
    pr_array4 = precisions[3, :, 0, 0, 2] 
    pr_array5 = precisions[4, :, 0, 0, 2] 
    pr_array6 = precisions[5, :, 0, 0, 2] 
    pr_array7 = precisions[6, :, 0, 0, 2] 
    pr_array8 = precisions[7, :, 0, 0, 2] 
    pr_array9 = precisions[8, :, 0, 0, 2] 
    pr_array10 = precisions[9, :, 0, 0, 2] 


    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(fontsize=18,loc="lower left")
    plt.show()
    plt.savefig('eval/pr/our.png')
if __name__ == "__main__":
    '''
    precise_our = get_precise(CONFIG_FILE , RESULT_FILE_OUR , metric="bbox")
    np.save('eval/pr/pkl/our_precise.npy',precise_our)
    '''
    precise_our = np.load('eval/pr/pkl/our_precise.npy')
    precise_mix = get_precise(CONFIG_FILE , RESULT_FILE_MIX, metric="bbox")
    precise_sf = get_precise(CONFIG_FILE , RESULT_FILE_SF, metric="bbox")
    precise_tar = get_precise(CONFIG_FILE , RESULT_FILE_TR, metric="bbox")
    precise_ad = get_precise(CONFIG_FILE , RESULT_FILE_AD, metric="bbox")
    
    #iou=0.5

    pr_array1 = np.mean(precise_our[0, :, :, 0, 2] ,axis=1)
    pr_array2 = np.mean(precise_mix[0, :, :, 0, 2] ,axis=1)
    pr_array3 = np.mean(precise_sf[0,:, :, 0, 2] ,axis=1)
    pr_array4 = np.mean(precise_tar[0, :, :, 0, 2] ,axis=1)
    pr_array5 = np.mean(precise_ad[0, :, :, 0, 2] ,axis=1)
    
    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 15,
            }

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1,linewidth=3.0, label="ours")
    plt.plot(x, pr_array2, label="mixing")
    plt.plot(x, pr_array3, label="sequential finetuning")
    plt.plot(x, pr_array4, label="target only ")
    plt.plot(x, pr_array5, label="adversarial learning")
    plt.xlabel("recall",font)
    plt.ylabel("precison",font)
    plt.xlim(0, 1.0 ,font)
    plt.ylim(0, 1.01 ,font)
    plt.grid(True)
    plt.legend(fontsize=13 , loc="lower left")
    plt.show()
    plt.savefig('eval/pr/iou=0.5.eps')
    