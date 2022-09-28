# -*- coding: utf-8 -*-
# @Time : 2022/9/28 14:50
# @Author : luff543
# @Email : luff543@gmail.com
# @File : train.py
# @Software: PyCharm

import argparse
from engines.utils.io import fold_check
from engines.utils.logger import get_logger
from engines.configure import Configure
import os

import numpy as np
import tensorflow as tf
import codecs
import pickle

from bert_base.bert import modeling
from bert_base.bert import tokenization
from collections import OrderedDict
from bert_base.train import conlleval
from bert_crf_multitask_model import CRFModel
from tensorflow.contrib.layers.python.layers import initializers


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None, seq_bieos_scheme=None, seq_type=None, tokenizer_text=None,
                 tokenizer_label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.seq_bieos_scheme = seq_bieos_scheme
        self.seq_type = seq_type
        self.tokenizer_text = tokenizer_text
        self.tokenizer_label = tokenizer_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, seq_bieos_scheme_ids, seq_type_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.seq_bieos_scheme_ids = seq_bieos_scheme_ids
        self.seq_type_ids = seq_type_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.seq_bieos_schemes = []
        self.seq_types = []
        self.labels = []
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        if len(self.labels) > 0:
            self.labels.append("X")
            self.labels.append("[CLS]")
            self.labels.append("[SEP]")
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                pickle.dump(self.labels, rf)
        return self.labels

    def get_seq_bieos_schemes(self, seq_bieos_schemes=None):
        if len(self.seq_bieos_schemes) > 0:
            self.seq_bieos_schemes.append("X")
            self.seq_bieos_schemes.append("[CLS]")
            self.seq_bieos_schemes.append("[SEP]")
            with codecs.open(os.path.join(self.output_dir, 'seq_bieos_schemes_list.pkl'), 'wb') as rf:
                pickle.dump(self.seq_bieos_schemes, rf)
        return self.seq_bieos_schemes

    def get_seq_types(self, seq_types=None):
        if len(self.seq_types) > 0:
            self.seq_types.append("X")
            self.seq_types.append("[CLS]")
            self.seq_types.append("[SEP]")
            with codecs.open(os.path.join(self.output_dir, 'seq_types_list.pkl'), 'wb') as rf:
                pickle.dump(self.seq_types, rf)
        return self.seq_types

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            seq_bieos_scheme = tokenization.convert_to_unicode(line[0])
            seq_type = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            text = tokenization.convert_to_unicode(line[3])

            examples.append(
                InputExample(guid=guid, text=text, label=label, seq_bieos_scheme=seq_bieos_scheme, seq_type=seq_type))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            seq_bieos_schemes = []
            seq_types = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    if tokens[-1] == '0':
                        labels.append('O')
                    else:
                        labels.append(tokens[-1])
                        if tokens[-1] == 'O':
                            seq_bieos_schemes.append('O')
                            seq_types.append('O')
                        elif tokens[-1].find('-'):
                            split_label = tokens[-1].split('-')
                            seq_bieos_schemes.append(split_label[0])
                            seq_types.append(split_label[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        seq_bieos_scheme = []
                        seq_type = []
                        label = []
                        word = []
                        for bieos_scheme, type, l, w in zip(seq_bieos_schemes, seq_types, labels, words):
                            if len(bieos_scheme) > 0 and len(type) and len(l) > 0 and len(w) > 0:
                                seq_bieos_scheme.append(bieos_scheme)
                                seq_type.append(type)
                                label.append(l)
                                if bieos_scheme not in self.seq_bieos_schemes:
                                    self.seq_bieos_schemes.append(bieos_scheme)

                                if type not in self.seq_types:
                                    self.seq_types.append(type)

                                if l not in self.labels:
                                    self.labels.append(l)
                                word.append(w)
                        lines.append([' '.join(seq_bieos_scheme), ' '.join(seq_type), ' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        seq_bieos_schemes = []
                        seq_types = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines


def write_tokens(tokens, output_dir, mode):
    """
    Write sequence parsing results to a file
    Only enabled when mode = test
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, bieos_scheme_list, type_list, max_seq_length, tokenizer,
                           output_dir, mode):
    """Analyze a sample, then convert words into ids, tags into ids, and then structure them into InputFeatures instances
    Args:
        ex_index: example index
        example: a sample
        label_list: label list
        bieos_scheme_list: boundary label list
        type_list: type list
        max_seq_length:
        tokenizer:
        output_dir
        mode:

    Return:
        structured InputFeatures instances
    """
    # label_list
    label_map = {}
    # 1 means to index the label from 1
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # Save the map of label->index
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    # bieos_scheme_list
    bieos_scheme_map = {}
    # 1 means indexing bieos_scheme from 1
    for (i, bieos_scheme) in enumerate(bieos_scheme_list, 1):
        bieos_scheme_map[bieos_scheme] = i
    # map of bieos_scheme->index
    if not os.path.exists(os.path.join(output_dir, 'bieos_scheme2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'bieos_scheme2id.pkl'), 'wb') as w:
            pickle.dump(bieos_scheme_map, w)

    # type_list
    type_map = {}
    # 0 means to index the type from 0
    for (i, type) in enumerate(type_list, 0):
        type_map[type] = i
    # Save the map of type->index
    if not os.path.exists(os.path.join(output_dir, 'type2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'type2id.pkl'), 'wb') as w:
            pickle.dump(type_map, w)

    text_list = example.text.split(' ')
    label_list = example.label.split(' ')
    bieos_scheme_list = example.seq_bieos_scheme.split(' ')
    type_list = example.seq_type.split(' ')

    tokens = []
    labels = []
    bieos_schemes = []
    types = []

    tokenizer_text = ""
    tokenizer_label = ""
    for i, word in enumerate(text_list):
        # Word segmentation, if it is Chinese, it is word segmentation,
        # but some characters that are not in BERT's vocab.txt will be processed by WordPice
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_list[i]
        bieos_scheme_1 = bieos_scheme_list[i]
        type_1 = type_list[i]
        for m in range(len(token)):
            tokenizer_text += token[m] + ' '
            if m == 0:
                if label_1 == 'O':
                    labels.append(label_1)
                    bieos_schemes.append(bieos_scheme_1)
                    types.append(type_1)
                    tokenizer_label += label_1 + ' '
                elif label_1.find('-'):
                    split_label_1 = label_1.split("-")
                    if (split_label_1[0] == "E" and m != (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "S" and len(token) >= 2):
                        labels.append("B" + "-" + split_label_1[1])
                        bieos_schemes.append("B")
                        types.append(type_1)
                        tokenizer_label += "B" + "-" + type_1 + " "
                    else:
                        labels.append(label_1)
                        bieos_schemes.append(bieos_scheme_1)
                        types.append(type_1)
                        tokenizer_label += label_1 + " "
            else:
                if label_1 == 'O':
                    labels.append(label_1)
                    bieos_schemes.append(label_1)
                    types.append(label_1)
                    tokenizer_label += label_1 + " "
                elif label_1.find('-'):
                    split_label_1 = label_1.split("-")
                    if (split_label_1[0] == "B" and m != (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "B" and m == (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "S" and m != (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "S" and m == (len(token) - 1)):
                        labels.append("E" + "-" + split_label_1[1])
                        bieos_schemes.append("E")
                        types.append(type_1)
                        tokenizer_label += "E" + "-" + type_1 + " "
                    elif (split_label_1[0] == "I" and m != (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "I" and m == (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "E" and m != (len(token) - 1)):
                        labels.append("I" + "-" + split_label_1[1])
                        bieos_schemes.append("I")
                        types.append(type_1)
                        tokenizer_label += "I" + "-" + type_1 + " "
                    elif (split_label_1[0] == "E" and m == (len(token) - 1)):
                        labels.append("E" + "-" + split_label_1[1])
                        bieos_schemes.append("E")
                        types.append(type_1)
                        tokenizer_label += "E" + "-" + type_1 + " "
                    else:
                        labels.append("X")
                        bieos_schemes.append("X")
                        types.append("X")
                        tokenizer_label += "X" + " "
    # sequence truncation
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 The reason is because the sequence needs to add a sentence start and sentence end marker
        labels = labels[0:(max_seq_length - 2)]
        bieos_schemes = bieos_schemes[0:(max_seq_length - 2)]
        types = types[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    bieos_scheme_ids = []
    type_ids = []
    ntokens.append("[CLS]")  # Sentence start set CLS symbol
    segment_ids.append(0)

    label_ids.append(label_map["[CLS]"])
    bieos_scheme_ids.append(bieos_scheme_map["[CLS]"])
    type_ids.append(type_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        bieos_scheme_ids.append(bieos_scheme_map[bieos_schemes[i]])
        type_ids.append(type_map[types[i]])
    ntokens.append("[SEP]")  # Add [SEP] symbol at the end of the sentence
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    bieos_scheme_ids.append(bieos_scheme_map["[SEP]"])
    type_ids.append(type_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # Convert the words (ntokens) in the sequence to id form
    input_mask = [1] * len(input_ids)
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        bieos_scheme_ids.append(0)
        type_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(bieos_scheme_ids) == max_seq_length
    assert len(type_ids) == max_seq_length

    # Print some sample data information
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logger.info("bieos_scheme_ids: %s" % " ".join([str(x) for x in bieos_scheme_ids]))
        logger.info("type_ids: %s" % " ".join([str(x) for x in type_ids]))

    if (tokenizer_text.endswith(" ")):
        tokenizer_text = tokenizer_text[:-1]
    if (tokenizer_label.endswith(" ")):
        tokenizer_label = tokenizer_label[:-1]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        seq_bieos_scheme_ids=bieos_scheme_ids,
        seq_type_ids=type_ids
    )
    example.tokenizer_text = tokenizer_text
    example.tokenizer_label = tokenizer_label

    write_tokens(ntokens, output_dir, mode)
    return feature


def _get_mini_batch_start_end(n_data, batch_size=None):
    '''
    Args:
        n_train: int, number of training instances
        batch_size: int (or None if full batch)

    Returns:
        batches: list of tuples of (start, end) of each mini batch
    '''
    mini_batch_size = n_data if batch_size is None else batch_size
    batches = zip(
        range(0, n_data, mini_batch_size),
        list(range(mini_batch_size, n_data, mini_batch_size)) + [n_data]
    )
    return batches


def create_model(Model_class, config):
    model = Model_class(config)
    return model


# config for the model
def config_model(seq_length, num_labels, num_bieos_schemes, num_types, num_train_steps, num_warmup_steps,
                 bieos_scheme_o_label_index, type_o_label_index):
    config = OrderedDict()
    config["train_keep_prob"] = 1.0
    config["optimizer"] = "adam"
    config["learning_rate"] = 1e-5
    config["initializers"] = initializers
    config["seq_length"] = seq_length
    config["num_labels"] = num_labels
    config["num_bieos_schemes"] = num_bieos_schemes
    config["num_types"] = num_types
    config["is_training"] = False
    config["dropout_rate"] = 1.0
    config["clip"] = 5
    config["init_checkpoint"] = bert_workspace + "bert_model.ckpt"
    config["num_train_steps"] = num_train_steps
    config["num_warmup_steps"] = num_warmup_steps
    config["rnn_cell_type"] = "lstm"
    config["rnn_hidden_unit"] = 200
    config["rnn_num_layers"] = 1
    config["crf_only"] = False

    config["bieos_scheme_o_label_index"] = bieos_scheme_o_label_index
    config["type_o_label_index"] = type_o_label_index

    bert_config_file = bert_workspace + "bert_config.json"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    config["bert_config"] = bert_config

    config["input_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                         name="input_ids")
    config["input_mask"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                          name="input_mask")
    config["segment_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                           name="segment_ids")
    config["label_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                         name="label_ids")
    config["bieos_scheme_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                                name="bieos_scheme_ids")
    config["type_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, seq_length],
                                        name="type_ids")
    config["global_step"] = tf.Variable(0, trainable=False)

    config["dropout_keep_prob"] = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
    return config


def evaluate(sess, model, name, data, examples, bieos_scheme_id_to_tag, type_id_to_tag,
             batch_size):
    logger.info("evaluate:{}".format(name))
    input_ids = data[0]
    input_mask = data[1]
    segment_ids = data[2]

    ner_results = model.evaluate(sess, input_ids, input_mask, segment_ids,
                                 bieos_scheme_id_to_tag, type_id_to_tag, batch_size)

    output_predict_file = os.path.join(output_dir,
                                       "label_" + str(name) + ".txt")
    output_predict_debug_file = os.path.join(output_dir,
                                             "label_" + str(name) + "_debug.txt")

    def result_to_pair(writer):
        writer.write('token' + ' ' + "label" + ' ' + "predict" + '\n')
        for predict_line, prediction in zip(examples,
                                            ner_results):

            idx = 0
            line = ''
            predict_line_tokenizer_text = '[CLS]' + ' ' + str(predict_line.tokenizer_text) + ' ' + '[SEP]'
            predict_line_tokenizer_label = '[CLS]' + ' ' + str(predict_line.tokenizer_label) + ' ' + '[SEP]'
            line_token = predict_line_tokenizer_text.split(' ')
            label_token = predict_line_tokenizer_label.split(' ')
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                logger.info(predict_line.text)
                logger.info(predict_line.label)
                break

            for current_bieos_labels, current_type_labels in zip(prediction[0], prediction[1]):
                if idx >= len_seq:
                    break
                try:
                    line += line_token[idx] + ' ' + label_token[
                        idx] + ' ' + current_bieos_labels + '-' + current_type_labels + '\n'
                except Exception as e:
                    logger.info(e)
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    line = ''
                    break
                idx += 1
            writer.write(line + '\n')

    with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
        result_to_pair(writer)

    eval_result = conlleval.return_report(output_predict_file)
    logger.info(''.join(eval_result))

    with codecs.open(
            os.path.join(output_dir, 'predict_score_' + name + '.txt'), 'a',
            encoding='utf-8') as fd:
        fd.write(''.join(eval_result))


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def restore_training_task(save_model_dir):
    model_paths = []
    for dir_path, _, filenames in os.walk(save_model_dir):
        if filenames:
            for filename in filenames:
                if filename.endswith(".meta"):
                    model_name = os.path.splitext(filename)[0]
                    model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths = sorted(model_paths, key=lambda x: int(x.split("/")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("/")[-2])

    return restore_epoch, restore_model_path


def restore_training_model(sess, save_model_dir):
    model_paths = []
    for dir_path, _, filenames in os.walk(save_model_dir):
        if filenames:
            for filename in filenames:
                if filename.endswith(".meta"):
                    model_name = os.path.splitext(filename)[0]
                    model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths = sorted(model_paths, key=lambda x: int(x.split("/")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("/")[-2])
    imported_meta = tf.train.import_meta_graph(
        os.path.join(restore_model_path + '.meta'))
    imported_meta.restore(sess, restore_model_path)
    return restore_epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BERT CRF')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)
    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs.CUDA_VISIBLE_DEVICES)

    workspace = configs.workspace
    bert_workspace = configs.bert_workspace

    batch_size = configs.batch_size
    num_train_epochs = configs.epoch
    max_seq_length = configs.max_sequence_length

    logger.info("start")
    # limit GPU memory
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    gpu_options.allow_growth = True
    tf_config = tf.ConfigProto(gpu_options=gpu_options)

    vocab_file = bert_workspace + "vocab.txt"
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    #
    processors = {
        "ner": NerProcessor
    }
    ner = "ner"
    output_dir = workspace + configs.output_dir
    processor = processors[ner](output_dir)

    train_examples = None
    data_dir = workspace + configs.datasets_fold
    train_examples = processor.get_train_examples(data_dir)
    label_list = processor.get_labels()
    bieos_scheme_list = processor.get_seq_bieos_schemes()
    type_list = processor.get_seq_types()
    num_labels = len(label_list)
    num_bieos_schemes = len(bieos_scheme_list)
    num_types = len(type_list)

    valid_label_list = []
    if os.path.exists(os.path.join(output_dir, 'label_list.pkl')):
        with codecs.open(os.path.join(output_dir, 'label_list.pkl'), 'rb') as rf:
            valid_label_list = pickle.load(rf)

    # label
    label_map = {}
    label2id = {}
    id2label = {}

    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
        label2id[label] = i
        id2label[i] = label

    # Save the map of label->index
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    # bieos_scheme
    bieos_scheme_map = {}
    bieos_scheme2id = {}
    id2bieos_scheme = {}

    for (i, bieos_scheme) in enumerate(bieos_scheme_list, 1):
        bieos_scheme_map[bieos_scheme] = i
        bieos_scheme2id[bieos_scheme] = i
        id2bieos_scheme[i] = bieos_scheme

    # Save the map of bieos_scheme->index
    if not os.path.exists(os.path.join(output_dir, 'bieos_scheme2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'bieos_scheme2id.pkl'), 'wb') as w:
            pickle.dump(bieos_scheme_map, w)

    # type
    type_map = {}
    type2id = {}
    id2type = {}
    for (i, type) in enumerate(type_list, 0):
        type_map[type] = i
        type2id[type] = i
        id2type[i] = type

    # Save the map of type->index
    if not os.path.exists(os.path.join(output_dir, 'type2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'type2id.pkl'), 'wb') as w:
            pickle.dump(type_map, w)

    sample_size = len(train_examples)
    num_train_steps = int(
        len(train_examples) * 1.0 / batch_size * num_train_epochs)
    warmup_proportion = 0.1
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    mode = None

    # load training data
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_label_ids = []
    train_bieos_scheme_ids = []
    train_type_ids = []
    # Traverse training data
    for (ex_index, example) in enumerate(train_examples):
        # for each training sample
        feature = convert_single_example(ex_index, example, label_list, bieos_scheme_list, type_list,
                                         max_seq_length, tokenizer,
                                         output_dir, mode)
        train_input_ids.append(feature.input_ids)
        train_input_mask.append(feature.input_mask)
        train_segment_ids.append(feature.segment_ids)
        train_label_ids.append(feature.label_ids)
        train_bieos_scheme_ids.append(feature.seq_bieos_scheme_ids)
        train_type_ids.append(feature.seq_type_ids)


        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

    # load development data
    development_examples = processor.get_dev_examples(data_dir)
    dev_input_ids = []
    dev_input_mask = []
    dev_segment_ids = []
    dev_label_ids = []
    dev_bieos_scheme_ids = []
    dev_type_ids = []
    # Traverse development data
    for (ex_index, example) in enumerate(development_examples):
        # For each development sample
        feature = convert_single_example(ex_index, example, label_list, bieos_scheme_list, type_list,
                                         max_seq_length, tokenizer, output_dir,
                                         mode="test")

        dev_input_ids.append(feature.input_ids)
        dev_input_mask.append(feature.input_mask)
        dev_segment_ids.append(feature.segment_ids)
        dev_label_ids.append(feature.label_ids)
        dev_bieos_scheme_ids.append(feature.seq_bieos_scheme_ids)
        dev_type_ids.append(feature.seq_type_ids)

    dev_data = [dev_input_ids, dev_input_mask, dev_segment_ids, dev_label_ids, dev_bieos_scheme_ids, dev_type_ids]

    # load testing data
    predict_examples = processor.get_test_examples(data_dir)
    test_input_ids = []
    test_input_mask = []
    test_segment_ids = []
    test_label_ids = []
    test_bieos_scheme_ids = []
    test_type_ids = []
    # Traverse test data
    for (ex_index, example) in enumerate(predict_examples):
        # for each test sample
        feature = convert_single_example(ex_index, example, label_list, bieos_scheme_list, type_list,
                                         max_seq_length, tokenizer, output_dir,
                                         mode="test")

        test_input_ids.append(feature.input_ids)
        test_input_mask.append(feature.input_mask)
        test_segment_ids.append(feature.segment_ids)
        test_label_ids.append(feature.label_ids)
        test_bieos_scheme_ids.append(feature.seq_bieos_scheme_ids)
        test_type_ids.append(feature.seq_type_ids)

    test_data = [test_input_ids, test_input_mask, test_segment_ids, test_label_ids, test_bieos_scheme_ids,
                 test_type_ids]

    feed_input_ids = []
    feed_input_mask = []
    feed_segment_ids = []
    feed_label_ids = []

    is_save_model = True
    model_save_dir = workspace + configs.model_fold + "/"

    is_restore_model = False

    restore_epoch = 0
    restore_model_path = ""
    if is_restore_model:
        restore_epoch, restore_model_path = restore_training_task(model_save_dir)
    config = config_model(seq_length=max_seq_length, num_labels=num_labels, num_bieos_schemes=num_bieos_schemes,
                          num_types=num_types, num_train_steps=num_train_steps,
                          num_warmup_steps=num_warmup_steps, bieos_scheme_o_label_index=bieos_scheme2id["O"],
                          type_o_label_index=type2id["O"], )

    with tf.compat.v1.Session(
            config=tf_config) as sess:
        crfModel = create_model(CRFModel, config)
        if is_save_model:
            crfModel.save_model_init(model_save_dir)
        if is_restore_model:
            restore_epoch = restore_training_model(sess, model_save_dir)
        if is_restore_model == False:
            sess.run(tf.global_variables_initializer())

        for t in range(restore_epoch + 1, num_train_epochs + 1):
            total_loss = 0.0
            abs_total_loss = 0.0
            batch_loss_list = []
            abs_batch_loss_list = []
            for start_i in range(0, sample_size, batch_size):
                end_i = start_i + batch_size
                feed_input_ids = train_input_ids[start_i:end_i]
                feed_input_mask = train_input_mask[start_i:end_i]
                feed_segment_ids = train_segment_ids[start_i:end_i]
                feed_label_ids = train_label_ids[start_i:end_i]
                feed_bieos_scheme_ids = train_bieos_scheme_ids[start_i:end_i]
                feed_type_ids = train_type_ids[start_i:end_i]

                step, batch_loss = crfModel.run_step(sess, True, feed_input_ids, feed_input_mask, feed_segment_ids,
                                                     feed_label_ids, feed_bieos_scheme_ids, feed_type_ids)
                total_loss += batch_loss
                batch_loss_list.append(batch_loss)

                abs_total_loss += abs(batch_loss)
                abs_batch_loss_list.append(abs(batch_loss))
            logger.info("step: " + str(t))
            logger.info("total loss: " + str(total_loss) + ", mean loss: " + str(np.mean(batch_loss_list)))

            name = "dev_" + str(t)
            best = evaluate(sess, crfModel, name, dev_data, development_examples, id2bieos_scheme, id2type,
                            batch_size)

            name = "test_" + str(t)
            best = evaluate(sess, crfModel, name, test_data, predict_examples, id2bieos_scheme, id2type,
                            batch_size)

            if is_save_model:
                crfModel.save_model(sess, t)
