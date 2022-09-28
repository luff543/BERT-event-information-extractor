# -*- coding: utf-8 -*-
# @Time : 2022/9/28 14:50
# @Author : luff543
# @Email : luff543@gmail.com
# @File : bert_crf_multitask_model.py
# @Software: PyCharm

import os
import numpy as np
import tensorflow as tf
from bert_base.bert import modeling
from tensorflow.contrib.crf import viterbi_decode

from tensorflow.contrib import crf
from tensorflow.contrib import rnn
from bert_base.bert import optimization
import math


class CRFModel(object):
    def __init__(self, config):

        self.config = config
        self.dropout_rate = config["dropout_rate"]
        self.lr = config["learning_rate"]
        self.initializers = config["initializers"]
        self.num_labels = config["num_labels"]
        self.num_bieos_schemes = config["num_bieos_schemes"]
        self.num_types = config["num_types"]
        self.is_training = config["is_training"]
        self.bert_config = config["bert_config"]
        self.init_checkpoint = config["init_checkpoint"]
        self.num_train_steps = config["num_train_steps"]
        self.num_warmup_steps = config["num_warmup_steps"]
        self.rnn_hidden_unit = config["rnn_hidden_unit"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.rnn_num_layers = config["rnn_num_layers"]
        self.crf_only = config["crf_only"]

        self.input_ids = config["input_ids"]
        self.input_mask = config["input_mask"]
        self.segment_ids = config["segment_ids"]
        self.label_ids = config["label_ids"]
        self.bieos_scheme_ids = config["bieos_scheme_ids"]
        self.type_ids = config["type_ids"]
        self.bieos_scheme_o_label_index = config["bieos_scheme_o_label_index"]
        self.type_o_label_index = config["type_o_label_index"]
        self.dropout_keep_prob = config["dropout_keep_prob"]
        self.global_step = config["global_step"]

        # Load BertModel to get the corresponding word embedding
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # Get the corresponding embedded from input data [batch_size, seq_length, embedding_size = 768]
        embedding = model.get_sequence_output()
        self.embedding = embedding
        max_seq_length = embedding.shape[1].value
        # Calculate the real length of the sequence
        used = tf.sign(tf.abs(self.input_ids))
        # A vector of size batch_size, containing the length of the sequence in the current batch
        self.lengths = tf.reduce_sum(used,
                                     reduction_indices=1)
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]
        self.seq_length = embedding.shape[1].value

        self.embedding_dims = embedding.shape[-1].value
        self.embedded_chars = tf.nn.dropout(embedding, self.dropout_keep_prob)

        if self.crf_only:
            bieos_logits = self.project_bieos_layer(self.embedded_chars)
            self.bieos_logits = bieos_logits
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            # project
            bieos_logits = self.project_bilstm_bieos_layer(lstm_output)
            self.lstm_output = lstm_output
            self.bieos_logits = bieos_logits

        # crf
        log_likelihood, trans = self.crf_layer(bieos_logits)
        self.log_likelihood = log_likelihood

        self.trans = trans
        bieos_predicts, bieos_crf_scores = crf.crf_decode(potentials=bieos_logits, transition_params=trans,
                                                          sequence_length=self.lengths)
        self.bieos_predicts = bieos_predicts
        self.bieos_crf_scores = bieos_crf_scores
        #
        with tf.variable_scope('type_projection'):
            type_logits = tf.layers.dense(lstm_output, self.num_types)  # B * S * num_types
            type_probabilities = tf.nn.softmax(type_logits, axis=-1)
            type_predicts = tf.argmax(type_probabilities, axis=-1, name="type_predicts")  # B * S

            self.type_logits = type_logits
            self.type_probabilities = type_probabilities
            self.type_predicts = type_predicts

        with tf.variable_scope('loss'):
            bieos_loss = -log_likelihood

            type_id_one_hot = tf.one_hot(indices=self.type_ids, depth=self.num_types, axis=-1)
            self.type_id_one_hot = type_id_one_hot
            type_loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.type_logits,
                                                                              labels=type_id_one_hot)
            type_loss = tf.reduce_sum(type_loss_cross_entropy)
            loss = bieos_loss + type_loss
            self.type_loss_cross_entropy = type_loss_cross_entropy
            self.bieos_loss = bieos_loss
            self.type_loss = type_loss_cross_entropy
            self.loss = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        # Load the BERT model
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        self.train_op = optimization.create_optimizer(
            self.loss, self.lr, self.num_train_steps, self.num_warmup_steps,
            False)

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        """
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)

        Return:
            batches: list of tuples of (start, end) of each mini batch
        """
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches

    def create_feed_dict(self, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids,
                         feed_bieos_scheme_ids, feed_type_ids):
        """
        Args:
            is_train: Flag, True for train batch
            feed_input_ids/feed_input_mask/feed_segment_ids/
            feed_label_ids/feed_bieos_scheme_ids/feed_type_ids: list train/evaluate data

        Return:
            structured data to feed
        """
        feed_dict = {
            self.input_ids: feed_input_ids,
            self.input_mask: feed_input_mask,
            self.segment_ids: feed_segment_ids,
            self.dropout_keep_prob: 1.0
        }
        if is_train:
            feed_dict[self.label_ids] = feed_label_ids
            feed_dict[self.bieos_scheme_ids] = feed_bieos_scheme_ids
            feed_dict[self.type_ids] = feed_type_ids
            feed_dict[self.dropout_keep_prob] = self.config["train_keep_prob"]
        return feed_dict

    def run_step(self, sess, is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids=None,
                 feed_bieos_scheme_ids=None, feed_type_ids=None):
        """
        Args:
            sess: session to run the batch
            is_train: a flag indicate if it is a train batch
            feed_input_ids/feed_input_mask/feed_segment_ids/
            feed_label_ids/feed_bieos_scheme_ids/feed_type_ids: list train/evaluate data

        Returns:
            step
            loss: loss of the batch
        """
        feed_dict = self.create_feed_dict(is_train, feed_input_ids, feed_input_mask, feed_segment_ids, feed_label_ids,
                                          feed_bieos_scheme_ids, feed_type_ids)
        if is_train:
            global_step, loss, bieos_loss, type_loss, _ = sess.run(
                [self.global_step, self.loss, self.bieos_loss, self.type_loss, self.train_op],
                feed_dict)
            if math.isnan(loss):
                print("math.isnan(loss)=True")
            return global_step, loss
        else:
            lengths, bieos_logits, transition_params, bieos_predicts, type_probabilities, type_predicts = sess.run(
                [self.lengths, self.bieos_logits, self.trans, self.bieos_predicts, self.type_probabilities,
                 self.type_predicts], feed_dict)
            return lengths, bieos_logits, transition_params, bieos_predicts, type_probabilities, type_predicts

    def _decode(self, logits, lengths, transition_matrix):
        """
        Args:
            logits: [batch_size, num_steps, num_tags] float32, logits
            lengths: [batch_size]int32, real length of each sequence
            transition_matrix: transaction matrix for inference

        Return:
            paths: decode paths
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.config.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def decode(self, logits, lengths, transition_matrix):
        """
        Args:
            logits: [batch_size, num_steps, num_tags]float32, logits
            lengths: [batch_size]int32, real length of each sequence
            transition_matrix: transaction matrix for inference

        Returns:
            sequence results
            sequence result scores:
        """
        result_sequences = []
        result_sequence_scores = []
        for score, seq_len in zip(logits, lengths):
            score = score[:seq_len]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                score, transition_matrix)
            result_sequences.append(viterbi_sequence)
            result_sequence_scores.append(viterbi_score)
        return result_sequences, result_sequence_scores

    # def evaluate(self,
    #              sess,
    #              evaluate_input_ids,
    #              evaluate_input_mask,
    #              evaluate_segment_ids,
    #              evaluate_label_ids, evaluate_bieos_scheme_ids, evaluate_type_ids,
    #              label_id_to_tag, bieos_scheme_label_id_to_tag, type_label_id_to_tag,
    #              batch_size):
    def evaluate(self,
                 sess,
                 input_ids,
                 input_mask,
                 segment_ids, bieos_scheme_label_id_to_tag, type_label_id_to_tag,
                 batch_size):
        """
        Args:
            sess: session  to run the model
            input_ids/input_mask/segment_ids/bieos_scheme_label_id_to_tag/type_label_id_to_tag : list of data
            batch_size: int
        Return:
            evaluate result
        """
        results = []
        n_data = len(input_ids)
        batches = self._get_mini_batch_start_end(n_train=n_data, batch_size=batch_size)
        trans = self.trans.eval()  # tensor.eval() # Equivalent to sess.run(self.trans)
        for start, end in batches:
            feed_input_ids = input_ids[start:end]
            feed_input_mask = input_mask[start:end]
            feed_segment_ids = segment_ids[start:end]

            lengths, bieos_logits, transition_matrix, bieos_predicts, type_probabilities, type_predicts = self.run_step(
                sess, False,
                feed_input_ids,
                feed_input_mask,
                feed_segment_ids)

            batch_seq_bieos_paths, batch_seq_bieos_score = self.decode(bieos_logits, lengths, trans)
            j = 0
            for i in range(start, end):
                sentence_bieos_label = []
                sentence_type_label = []
                for k in range(0, len(batch_seq_bieos_paths[j])):
                    if batch_seq_bieos_paths[j][k] not in bieos_scheme_label_id_to_tag:
                        curr_bieos_label = "X"
                    else:
                        curr_bieos_label = bieos_scheme_label_id_to_tag[batch_seq_bieos_paths[j][k]]

                    if type_predicts[j][k] not in type_label_id_to_tag:
                        curr_type_label = "X"
                    else:
                        curr_type_label = type_label_id_to_tag[type_predicts[j][k]]

                    sentence_bieos_label.append(curr_bieos_label)
                    sentence_type_label.append(curr_type_label)
                j += 1
                results.append((sentence_bieos_label, sentence_type_label))
        return results

    def _witch_cell(self):
        """RNN type
        Return:
            RNN cell
        """
        cell_tmp = None
        if self.rnn_cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.rnn_hidden_unit)
        elif self.rnn_cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.rnn_hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """Bi-directional RNN
        Returns:
            cell forward
            cell backward
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """
        Return:
            Bidirectional LSTM output
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.rnn_num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.rnn_num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.rnn_num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_bieos_layer(self, lstm_outputs, name=None):
        """
        Args:
            lstm_outputs: [batch_size, num_steps, emb_size]
         Return:
            logits: [batch_size, num_steps, num_tags]
        """

        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.rnn_hidden_unit * 2, self.rnn_hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.rnn_hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.rnn_hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.rnn_hidden_unit, self.num_bieos_schemes],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_bieos_schemes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_bieos_schemes])

    def project_bieos_layer(self, embedding_chars, name=None):
        """
        Args:
            lstm_outputs: [batch_size, num_steps, emb_size]
         Return:
            boundary logits: [batch_size, num_steps, num_tags]
        """

        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_bieos_schemes],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_bieos_schemes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(embedding_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_bieos_schemes])

    def crf_layer(self, logits):
        """calculate crf loss
        Args:
            logits: [batch_size, num_steps, num_tags]
         Return:
            if train
               return log_likelihood, transition matrix
            else if test
               return None,  transition matrix
        """
        with tf.variable_scope("crf_layer"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_bieos_schemes, self.num_bieos_schemes],
                initializer=self.initializers.xavier_initializer())

            if self.bieos_scheme_ids is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.bieos_scheme_ids,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return log_likelihood, trans

    def save_model_init(self, model_save_dir):
        with tf.name_scope('save_model'):
            self.saver = tf.train.Saver(max_to_keep=0)
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
