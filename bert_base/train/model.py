# encoding = utf8
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.contrib import crf

from bert_base.bert import optimization
import datetime
import time as time
# import rnncell as rnn
# from utils import result_to_json
# from data_utils import create_input, iobes_iob

class CRFModel(object):
    def __init__(self, config):

        self.config = config
        # self.hidden_unit = config["hidden_unit"]
        self.dropout_rate = config["dropout_rate"]
        # self.cell_type = config["cell_type"]
        # self.num_layers = config["num_layers"]
        # self.embedded_chars = config["embedded_chars"]
        self.lr = config["learning_rate"]
        self.initializers = config["initializers"]
        #self.seq_length = config["seq_length"]
        self.num_labels = config["num_labels"]
        #self.lengths = config["lengths"]
        #self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = config["is_training"]
        self.bert_config = config["bert_config"]
        self.init_checkpoint = config["init_checkpoint"]
        self.num_train_steps = config["num_train_steps"]
        self.num_warmup_steps = config["num_warmup_steps"]
        # self.lr = config["lr"]
        # self.char_dim = config["char_dim"]
        # self.lstm_dim = config["lstm_dim"]
        # self.seg_dim = config["seg_dim"]
        #
        # self.num_chars = config["num_chars"]
        # self.num_segs = 4
        #
        # self.global_step = tf.Variable(0, trainable=False)
        # self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        # self.best_test_f1 = tf.Variable(0.0, trainable=False)
        # self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        # self.char_inputs = tf.placeholder(dtype=tf.int32,
        #                                   shape=[None, None],
        #                                   name="ChatInputs")
        # self.seg_inputs = tf.placeholder(dtype=tf.int32,
        #                                  shape=[None, None],
        #                                  name="SegInputs")
        #
        # self.targets = tf.placeholder(dtype=tf.int32,
        #                               shape=[None, None],
        #                               name="Targets")
        self.input_ids = config["input_ids"]
        self.input_mask = config["input_mask"]
        self.segment_ids = config["segment_ids"]
        #self.label_ids = config["label_ids"]
        #self.labels = config["label_ids"]
        self.label_ids = config["label_ids"]
        self.dropout_keep_prob = config["dropout_keep_prob"]
        self.global_step = config["global_step"]
        # input_ids = tf.placeholder(tf.int32, [None, 202], name="input_ids")
        # input_mask = tf.placeholder(tf.int32, [None, 202], name="input_mask")
        # segment_ids = tf.placeholder(tf.int32, [None, 202], name="segment_ids")
        # label_ids = tf.placeholder(tf.int32, [None, 202], name="label_ids")
        # dropout keep prob
        # self.dropout = tf.placeholder(dtype=tf.float32,
        #                               name="Dropout")
        # 使用数据加载BertModel,获取对应的字embedding
        import tensorflow as tf
        from bert_base.bert import modeling
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size=768]
        embedding = model.get_sequence_output()
        max_seq_length = embedding.shape[1].value
        # 算序列真实长度
        used = tf.sign(tf.abs(self.input_ids))
        self.lengths = tf.reduce_sum(used,
                                reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]
        #max_seq_length = embedding.shape[1].value
        self.seq_length = embedding.shape[1].value

        self.embedding_dims = embedding.shape[-1].value
        #if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            #self.embedded_chars = tf.nn.dropout(embedding, self.dropout_rate)
        # dropout keep prob
        # self.dropout = tf.placeholder(dtype=tf.float32,
        #                               name="Dropout")
        #self.embedded_chars = embedding
        self.embedded_chars = tf.nn.dropout(embedding, self.dropout_keep_prob)

        crf_only = True
        #if crf_only:
        logits = self.project_crf_layer(self.embedded_chars)
        #logits = self.project_crf_layer(self.embedded_chars)
        self.logits = logits

        # crf
        loss, trans = self.crf_layer(logits)
        self.loss = loss
        self.trans = trans
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                     sequence_length=self.lengths)
        self.pred_ids = pred_ids


        tvars = tf.trainable_variables()
        # 加载BERT模型
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        self.train_op = optimization.create_optimizer(
            self.loss, self.lr, self.num_train_steps, self.num_warmup_steps,
            False)
        #
        # with tf.variable_scope("optimizer"):
        #     optimizer = self.config["optimizer"]
        #     if optimizer == "sgd":
        #         self.opt = tf.train.GradientDescentOptimizer(self.lr)
        #     elif optimizer == "adam":
        #         self.opt = tf.train.AdamOptimizer(self.lr)
        #     elif optimizer == "adgrad":
        #         self.opt = tf.train.AdagradOptimizer(self.lr)
        #     else:
        #         raise KeyError

            # apply grad clip to avoid gradient explosion
            # grads_vars = self.opt.compute_gradients(self.loss)
            # capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                      for g, v in grads_vars]
            # self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
            # grads_and_vars = self.opt.compute_gradients(self.loss)
            # grads_and_vars = filter(lambda x: x[0] is not None, grads_and_vars)
            # grads_and_vars = [(tf.clip_by_norm(g, self.config["clip"]), v) for
            #                   g, v in grads_and_vars]
            # nil_grads_and_vars = []
            # for g, v in grads_and_vars:
            #     if v.name in self._nil_vars:
            #         nil_grads_and_vars.append((zero_nil_slot(g), v))
            #     else:
            #         nil_grads_and_vars.append((g, v))
            # self.train_op = self._opt.apply_gradients(grads_and_vars=nil_grads_and_vars,
            #                                                   global_step=self.global_step,
            #                                                   name="train_op")

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        '''
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)

        Returns:
            batches: list of tuples of (start, end) of each mini batch
        '''
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches

    def create_feed_dict(self, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        # _, chars, segs, tags = batch
        # feed_dict = {
        #     self.char_inputs: np.asarray(chars),
        #     self.seg_inputs: np.asarray(segs),
        #     self.dropout: 1.0,
        # }
        feed_dict = {
            self.input_ids: feed_input_ids,
            self.input_mask: feed_input_mask,
            self.segment_ids: feed_segment_ids,
            self.dropout_keep_prob: 1.0
        }
        if is_train:
            feed_dict[self.label_ids] = feed_label_ids
            feed_dict[self.dropout_keep_prob] = self.config["train_keep_prob"]
            #feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids=None):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            # lengths, logits, transition_params = sess.run([self.lengths, self.logits, self.trans], feed_dict)
            # return lengths, logits, transition_params
            lengths, logits, transition_params, pred_ids = sess.run(
                [self.lengths, self.logits, self.trans, self.pred_ids], feed_dict)
            return lengths, logits, transition_params, pred_ids
            #self.pred_ids = pred_ids
            # lengths, logits, predict_ids = sess.run(
            #     [self.lengths, self.logits, self.pred_ids], feed_dict)
            # return lengths, logits, predict_ids

    def _decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.config.num_tags + [0]])  # 初始化一个
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])  # 创建一个字符长度是 输入字长度维度元素为1的np数组
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def decode(self, logits, lengths, transition_matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        # paths = []
        # small = -1000.0
        # #start = np.asarray([[small]*self.num_tags +[0]])
        # #start = np.asarray([[small]*self.num_labels +[0]])
        # start = np.asarray([[small] * self.num_labels + [0]])
        # for score, length in zip(logits, lengths):
        #     score = score[:length]
        #     #pad = small * np.ones([length, 1])
        #     #logits = np.concatenate([score, pad], axis=1)
        #     #logits = np.concatenate([start, logits], axis=0)
        #     #path, _ = viterbi_decode(logits, matrix)
        #     path, _ = viterbi_decode(score, matrix)
        #
        #     paths.append(path[1:])
        result_sequences = []
        for score, seq_len in zip(logits, lengths):
            score = score[:seq_len]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                score, transition_matrix)
            result_sequences.append(viterbi_sequence)
        return result_sequences
        # return paths

    # def evaluate(self, sess, data_manager, id_to_tag):
    def evaluate(self,
                sess,
                evaluate_input_ids,
                evaluate_input_mask,
                evaluate_segment_ids,
                evaluate_label_ids,
                id_to_tag,
                batch_size):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        #evaluate_input_ids, evaluate_input_mask, evaluate_segment_ids, evaluate_label_ids = data
        results = []
        # trans = self.trans.eval() # tensor.eval() 相当于 sess.run(self.trans)作用；其实就是执行
        # sample_size = 72
        # for start_i in range(0, sample_size, batch_size):
        #     end_i = start_i + batch_size
        #     feed_input_ids = evaluate_input_ids[start_i:end_i]
        #     feed_input_mask = evaluate_input_mask[start_i:end_i]
        #     feed_segment_ids = evaluate_segment_ids[start_i:end_i]
        #     feed_label_ids = evaluate_label_ids[start_i:end_i]
        #     lengths, scores = self.run_step(sess, False, feed_input_ids,
        #                                     feed_input_mask, feed_segment_ids)
        #     batch_paths = self.decode(scores, lengths, trans)
        n_data = len(evaluate_input_ids)
        # n_data = 64
        batches = self._get_mini_batch_start_end(n_train=n_data, batch_size=batch_size)
        # unary_scores, transition_params, sentence_lens = [], None, []
        # predict_ids_results = []
        # gold_ids_results = []
        # predict_lengths = []
        # predict_labels_list = [] # 以sentence 為單位
        trans = self.trans.eval()  # tensor.eval() 相当于 sess.run(self.trans)作用；其实就是执行
        for start, end in batches:
            #print(str(start) + "-" + str(end))
            feed_input_ids = evaluate_input_ids[start:end]
            feed_input_mask = evaluate_input_mask[start:end]
            feed_segment_ids = evaluate_segment_ids[start:end]
            feed_label_ids = evaluate_label_ids[start:end]
            # lengths, scores, transition_matrix = self.run_step(sess, False, feed_input_ids,
            #                                 feed_input_mask, feed_segment_ids)
            # lengths, scores, transition_matrix = self.run_step(sess, False,
            #                                                    feed_input_ids,
            #                                                    feed_input_mask,
            #                                                    feed_segment_ids)
            lengths, scores, transition_matrix, pred_ids = self.run_step(sess, False,
                                                               feed_input_ids,
                                                               feed_input_mask,
                                                               feed_segment_ids)
            # lengths, scores, predict_ids = self.run_step(sess, False,
            #                                                    feed_input_ids,
            #                                                    feed_input_mask,
            #                                                    feed_segment_ids)
            # ner_results = []
            # for predict_id, seq_len in zip(predict_ids, lengths):
            #     predict_id = predict_id[:seq_len]
            #     ner_results.append(predict_id)
            #batch_paths = self._decode(scores, lengths, transition_matrix)
            #batch_paths = self.decode(scores, lengths, transition_matrix)
            batch_paths = self.decode(scores, lengths, trans)
            j = 0
            for i in range(start, end):
                predict_sentence_label = []
                evaluate_sentence_label = []
                for k in range(0, len(batch_paths[j])):
                    if batch_paths[j][k] not in id_to_tag:
                        #curr_label = str(batch_paths[j][k]) + " :key-error"
                        curr_label = "X"
                    else:
                        curr_label = id_to_tag[batch_paths[j][k]]
                    evaluate_sentence_label.append(curr_label)
                j += 1
            #results.extend(evaluate_sentence_label)
                results.append(evaluate_sentence_label)
            # j = 0
            # for i in range(start, end):
            #     sentence_length = predict_lengths[j]
            #     #print(i)
            #     pred_sentence_ids = pred_ids_result[j, 0:sentence_length]
            #     predict_ids_results.extend(pred_sentence_ids)
            #     #test_label_sentence_ids = test_label_batch_ids[j, 0:predict_lengths[j]]
            #     #test_label_sentence_ids = test_label_batch_ids[j]
            #     #test_sentence_label_ids = []
            #     #test_sentence_label = []
            #     predict_sentence_label = []
            #     for k in range(0, sentence_length):
            #         #test_sentence_label.append(id2label[test_label_batch_ids[j][k]])
            #         predict_sentence_label.append(id2label[pred_sentence_ids[k]])
            #     #predict_labels_list.append(test_sentence_label)
            #     predict_labels_list.append(predict_sentence_label)
            #     #gold_ids_results.extend(test_sentence_label_ids)
            #     # gold_sentence_ids = test_label_ids[i, 0:predict_lengths[j]]
            #     j += 1
        # for batch in data_manager.iter_batch():
        #     strings = batch[0]
        #     tags = batch[-1]
        #     lengths, scores = self.run_step(sess, False, batch)
        #     batch_paths = self.decode(scores, lengths, trans)
        #     for i in range(len(strings)):
        #         result = []
        #         string = strings[i][:lengths[i]]
        #         gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
        #         pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
        #         for char, gold, pred in zip(string, gold, pred):
        #             result.append(" ".join([char, gold, pred]))
        #         results.append(result)
        return results

    # def evaluate_line(self, sess, inputs, id_to_tag):
    #     trans = self.trans.eval()
    #     lengths, scores = self.run_step(sess, False, inputs)
    #     batch_paths = self.decode(scores, lengths, trans)
    #     tags = [id_to_tag[idx] for idx in batch_paths[0]]
    #     return result_to_json(inputs[0][0], tags)

#
    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size=768]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            # if self.labels is None:
            #     return None, trans
            if self.label_ids is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    # tag_indices=self.labels,
                    tag_indices=self.label_ids,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans

    def save_model_init(self, model_save_dir):
        with tf.name_scope('save_model'):
            #localtime = time.strftime("%Y_%m_%d", time.localtime())
            #self.saver = tf.train.Saver(max_to_keep=0)
            self.saver = tf.train.Saver(max_to_keep=5)
            #model_save_dir = model_dir + localtime + '/'
            self.model_save_dir = model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

    def save_model(self, sess, current_train_epoch):
        model_save_dir = self.model_save_dir + str(current_train_epoch) + '/'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            save_path = self.saver.save(sess, model_save_dir + 'model.ckpt')

    def restore_model(self, sess, model_file):
        self.saver.restore(sess, model_file)