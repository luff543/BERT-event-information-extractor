#! usr/bin/env python3
# -*- coding:utf-8 -*-

import collections
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
import codecs
import pickle
from bert_base.train.models import create_model, InputFeatures, InputExample
from bert_base.server.helper import set_logger

from bert_base.train import tf_metrics
from bert_base.bert import modeling
from bert_base.bert import optimization
from bert_base.bert import tokenization
from collections import OrderedDict
from tensorflow.contrib import crf
from bert_base.train import conlleval
from model import CRFModel
import datetime
import time as time

from tensorflow.contrib.layers.python.layers import initializers

logger = set_logger('NER Training')

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

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
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
                    # labels.append(tokens[-1])
                    if tokens[-1] == '0' :
                        labels.append('O')
                    else:
                        labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                if l not in self.labels:
                                    self.labels.append(l)
                                # self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines

def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
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

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
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
def config_model(seq_length, num_labels, num_train_steps, num_warmup_steps):
    config = OrderedDict()
    config["train_keep_prob"] = 0.8
    config["optimizer"] = "adam"
    config["learning_rate"] = 1e-5
    config["initializers"] = initializers
    config["seq_length"] = seq_length
    config["num_labels"] = num_labels
    config["is_training"] = False
    config["dropout_rate"] = 1.0
    config["clip"] = 5
    config["init_checkpoint"] = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV2/chinese_L-12_H-768_A-12/bert_model.ckpt"
    config["num_train_steps"] = num_train_steps
    config["num_warmup_steps"] = num_warmup_steps


    #custom define
    bert_config_file = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV2/chinese_L-12_H-768_A-12/bert_config.json"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    config["bert_config"] = bert_config

    config["input_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, 202],
                                    name="input_ids")
    config["input_mask"] = tf.placeholder(dtype=tf.int32, shape=[None, 202],
                                     name="input_mask")
    config["segment_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, 202],
                                      name="segment_ids")
    config["label_ids"] = tf.placeholder(dtype=tf.int32, shape=[None, 202],
                                    name="label_ids")
    config["global_step"] = tf.Variable(0, trainable=False)

    config["dropout_keep_prob"] = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
    return config

def evaluate(sess, model, name, data, evaluate_examples, id_to_tag, batch_size):
    logger.info("evaluate:{}".format(name))
    evaluate_input_ids = data[0]
    evaluate_input_mask = data[1]
    evaluate_segment_ids = data[2]
    evaluate_label_ids = data[3]
    ner_results = model.evaluate(sess, evaluate_input_ids, evaluate_input_mask, evaluate_segment_ids, evaluate_label_ids, id_to_tag, batch_size)

    # name = train_11
    # name = test_11
    output_predict_file = os.path.join(output_dir,
                                       "label_" + str(name) + ".txt")

    def result_to_pair(writer):
        for predict_line, prediction in zip(evaluate_examples,
                                            ner_results):
            idx = 0
            line = ''
            predict_line_text = '[CLS]' + ' ' + str(predict_line.text) + ' ' + '[SEP]'
            predict_line_tabel = '[CLS]' + ' ' + str(predict_line.label) + ' ' + '[SEP]'
            line_token = predict_line_text.split(' ')
            label_token = predict_line_tabel.split(' ')
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                logger.info(predict_line.text)
                logger.info(predict_line.label)
                break
            for curr_labels in prediction:
                if idx >= len_seq:
                    break
                # if id == 0:
                #     continue
                # curr_labels = id2label[id]
                # if curr_labels in ['[CLS]', '[SEP]']:
                #     continue
                try:
                    line += line_token[idx] + ' ' + label_token[
                        idx] + ' ' + curr_labels + '\n'
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
    print(''.join(eval_result))
    # 写结果到文件中
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
    for dir_path,_,filenames in os.walk(save_model_dir):
            if filenames:
                for filename in filenames:
                    if filename.endswith(".meta"):
                        model_name=os.path.splitext(filename)[0]
                        model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths=sorted(model_paths, key=lambda x: int(x.split("\\")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("\\")[-2])

    return restore_epoch, restore_model_path

def restore_training_model(sess, save_model_dir):
    model_paths = []
    for dir_path,_,filenames in os.walk(save_model_dir):
            if filenames:
                for filename in filenames:
                    if filename.endswith(".meta"):
                        model_name=os.path.splitext(filename)[0]
                        model_paths.append(os.path.abspath(os.path.join(dir_path, model_name)))
    model_paths=sorted(model_paths, key=lambda x: int(x.split("\\")[-2]))
    restore_model_path = model_paths[-1]
    restore_epoch = int(model_paths[-1].split("\\")[-2])
    imported_meta = tf.train.import_meta_graph(
        os.path.join(restore_model_path + '.meta'))
    imported_meta.restore(sess, restore_model_path)
    return restore_epoch

if __name__ == "__main__":
    print("start")
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    #is_training = True
    vocab_file = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV2/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    #
    processors = {
        "ner": NerProcessor
    }
    ner = "ner"
    output_dir = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV17/training"
    processor = processors[ner](output_dir)

    num_train_epochs = 30

    train_examples = None
    data_dir = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV17/NERdata"
    train_examples = processor.get_train_examples(data_dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    valid_label_list = []
    if os.path.exists(os.path.join(output_dir, 'label_list.pkl')):
        with codecs.open(os.path.join(output_dir, 'label_list.pkl'), 'rb') as rf:
            valid_label_list = pickle.load(rf)

    label_map = {}
    label2id = {}
    id2label = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
        label2id[label] = i
        id2label[i] = label

    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    sample_size = len(train_examples)
    batch_size = 8
    num_train_steps = int(
        len(train_examples) * 1.0 / batch_size * num_train_epochs)
    warmup_proportion = 0.1
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    max_seq_length = 202
    mode = None

    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_label_ids = []
    # 遍历训练数据
    for (ex_index, example) in enumerate(train_examples):
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer,
                                         output_dir, mode)
        train_input_ids.append(feature.input_ids)
        train_input_mask.append(feature.input_mask)
        train_segment_ids.append(feature.segment_ids)
        train_label_ids.append(feature.label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

    # load testing data
    predict_examples = processor.get_test_examples(data_dir)
    test_input_ids = []
    test_input_mask = []
    test_segment_ids = []
    test_label_ids = []
    # 遍历測試数据
    for (ex_index, example) in enumerate(predict_examples):
        # 对于每一个測試样本,
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, output_dir,
                                         mode="test")

        test_input_ids.append(feature.input_ids)
        test_input_mask.append(feature.input_mask)
        test_segment_ids.append(feature.segment_ids)
        test_label_ids.append(feature.label_ids)

    feed_input_ids = []
    feed_input_mask = []
    feed_segment_ids = []
    feed_label_ids = []

    train_data = [train_input_ids, train_input_mask, train_segment_ids,
                  train_label_ids]
    test_data = [test_input_ids, test_input_mask, test_segment_ids, test_label_ids]

    model_dir = "/home/luff543/BERT-BiLSTM-CRF-NER-ENV17/model/"
    is_save_model = True
    localtime = time.strftime("%Y_%m_%d", time.localtime())
    #model_save_dir = model_dir + localtime + '/'
    model_save_dir = model_dir + "2020_02_12" + "/"

    is_restore_model = False

    restore_epoch = 0
    restore_model_path = ""
    if is_restore_model:
        restore_epoch, restore_model_path = restore_training_task(model_save_dir)
    config = config_model(seq_length= 202, num_labels=num_labels, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
    with tf.Session(config=tf_config) as sess:
        crfModel = create_model(CRFModel, config)
        if is_save_model:
            crfModel.save_model_init(model_save_dir)
        if is_restore_model:
            #crfModel.restore_model(sess, restore_model_path)
            restore_epoch = restore_training_model(sess, model_save_dir)
        #if is_restore_model == False:
        if is_restore_model == False:
            sess.run(tf.global_variables_initializer())

        for t in range(restore_epoch + 1, num_train_epochs):
            total_loss = 0.0
            #sample_size = 72
            batch_loss_list = []
            for start_i in range(0, sample_size, batch_size):
                end_i = start_i + batch_size
                feed_input_ids = train_input_ids[start_i:end_i]
                feed_input_mask = train_input_mask[start_i:end_i]
                feed_segment_ids = train_segment_ids[start_i:end_i]
                feed_label_ids = train_label_ids[start_i:end_i]
                step, batch_loss = crfModel.run_step(sess, True, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids)
                total_loss += batch_loss
                batch_loss_list.append(batch_loss)
            print("step: " + str(step))
            print("total loss: " + str(total_loss) + ", mean loss: " + str(np.mean(batch_loss_list)))

            name = "train_" + str(t)
            best = evaluate(sess, crfModel, name, train_data, train_examples, id2label, batch_size)

            name = "test_" + str(t)
            best = evaluate(sess, crfModel, name, test_data, predict_examples, id2label, batch_size)
            # best = evaluate(sess, crfModel, name, train_data, train_examples,
            #                 id2label, batch_size)
            if is_save_model:
                crfModel.save_model(sess, t)
