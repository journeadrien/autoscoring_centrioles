# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:16:44 2020

@author: journe
"""

import subprocess
import sys
import os.path as op
import os
import time

EXPERIMENT_PATH = "E:\\Adrien\\data\\Experiment\\"

ANNOTATION_PATH = "E:\\Adrien\\data\\Annotation\\"

if __name__ == '__main__':
    exp_name = sys.argv[1]
    if exp_name == 'all':
        exp_list = next(os.walk(EXPERIMENT_PATH))[1]
    else:
        exp_list = [exp_name]
    for exp_name in exp_list:
        print('----- Starting analysis for exp: '+ exp_name)
        start = time.time()
        # subprocess.call(
        #     [
        #         'python',
        #         './nuclei_segmentation/predict.py',
        #         op.join(EXPERIMENT_PATH, exp_name),
        #         op.join(ANNOTATION_PATH, exp_name)
        #     ]
        # )
        # subprocess.call(
        #     [
        #         'python',
        #         './cell_segmentation/predict.py',
        #         op.join(EXPERIMENT_PATH, exp_name),
        #         op.join(ANNOTATION_PATH, exp_name)
        #     ]
        # )
        subprocess.call(
            [
                'python',
                './centriole_segmentation/predict.py',
                op.join(EXPERIMENT_PATH, exp_name),
                op.join(ANNOTATION_PATH, exp_name)
            ]
        )
        subprocess.call(
            [
                'python',
                './post_analysis/predict.py',
                op.join(EXPERIMENT_PATH, exp_name),
                op.join(ANNOTATION_PATH, exp_name)
            ]
        )
        print('----- Analysis Done for exp: {} and took {}'.format(exp_name,time.strftime("%H:%M:%S",time.gmtime(time.time()-start))))
    