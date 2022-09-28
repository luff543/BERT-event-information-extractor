# -*- coding: utf-8 -*-
# @Time : 2022/8/26 12:09
# @Author : luff543
# @Email : luff543@gmail.com
# @File : configure.py
# @Software: PyCharm

import sys


class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        the_item = 'workspace'
        if the_item in config:
            self.workspace = config[the_item]
        the_item = 'bert_workspace'
        if the_item in config:
            self.bert_workspace = config[the_item]
        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]
        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]
        the_item = 'output_dir'
        if the_item in config:
            self.output_dir = config[the_item]



        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])
        if self.max_sequence_length > 512:
            raise Exception('the max sequence length over 512 in Bert mode')
        
        the_item = 'CUDA_VISIBLE_DEVICES'
        if the_item in config:
            self.CUDA_VISIBLE_DEVICES = int(config[the_item])

        the_item = 'model_fold'
        if the_item in config:
            self.model_fold = config[the_item]


    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, 'r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                # noinspection PyBroadException
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info(' Datasets:')
        logger.info('     workspace            : {}'.format(self.workspace))
        logger.info('     bert_workspace       : {}'.format(self.bert_workspace))
        logger.info('     datasets         fold: {}'.format(self.datasets_fold))
        logger.info('     log               dir: {}'.format(self.log_dir))
        logger.info('     output            dir: {}'.format(self.output_dir))
        logger.info(' ' + '++' * 20)
        logger.info('Model Configuration:')
        logger.info('     max  sequence  length: {}'.format(self.max_sequence_length))
        logger.info('     CUDA  VISIBLE  DEVICE: {}'.format(self.CUDA_VISIBLE_DEVICES))
        logger.info(' ' + '++' * 20)
        logger.info(' Training Settings:')
        logger.info('     epoch                : {}'.format(self.epoch))
        logger.info('     batch            size: {}'.format(self.batch_size))
        logger.info('     model            fold: {}'.format(self.model_fold))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()
