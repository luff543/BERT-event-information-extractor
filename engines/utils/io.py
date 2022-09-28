# -*- coding: utf-8 -*-
# @Time : 2022/8/26 12:52
# @Author : luff543
# @Email : luff543@gmail.com
# @File : io.py
# @Software: PyCharm

import os


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    model_fold = 'model_fold'
    if not os.path.exists(configures.model_fold) or not hasattr(configures, model_fold):
        print('model_fold fold not found, creating...')
        paths = configures.model_fold.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(configures.model_folder):
            os.mkdir(configures.model_fold)
        elif not os.path.exists('model'):
            os.mkdir('model')

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(configures.log_dir)

    output_dir = 'output_dir'
    if not os.path.exists(configures.output_dir):
        print('output fold not found, creating...')
        if hasattr(configures, output_dir):
            os.mkdir(configures.output_dir)

