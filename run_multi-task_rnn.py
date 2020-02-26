# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import logging
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import multi_task_model

import subprocess
import stat

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 3000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 0,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", None, "Options: joint; intent; tagging")
tf.app.flags.DEFINE_string("embedding_path", None, "embedding array's path.")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    logging.info('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    logging.info('Please indicate task to run. Available options: intent; tagging; joint')
    exit()

task = dict({'intent':0, 'tagging':0, 'joint':0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1
    
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]
#_buckets = [(3, 10), (10, 25)]

# metrics function using conlleval.pl
def intenteval(p, filename, mode):
  """ This function is used to save the intent detection predicted """

  if mode == 'Test':
    path = FLAGS.data_dir + '/test/test.label'
  else:
    path = FLAGS.data_dir + '/valid/valid.label'
    
  with open(path) as infile:
    intents = [line.strip() for line in infile.read().split('\n')]

  out = ''
  for intent, pred in zip(intents, p):
    out += '{} {}\n'.format(intent, pred)

  with open(filename, 'w') as outfile:
    outfile.writelines(out[:-1]) # remove the ending \n on last line

def conlleval(p, g, w, filename, mode):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    # mode: Eval, Test
    if mode == 'Test':
      path_in = FLAGS.data_dir + '/test/test.seq.in'
      path_out = FLAGS.data_dir + '/test/test.seq.out'
    else:
      path_in = FLAGS.data_dir + '/valid/valid.seq.in'
      path_out = FLAGS.data_dir + '/valid/valid.seq.out'

    with open(path_in) as infile:
      utterances = [line.split() for line in infile.read().split('\n')]

    with open(path_out) as infile:
      slots = [line.split() for line in infile.read().split('\n')]

    out = ''

    for words, tags, pred in zip(utterances, slots, p):
      out += 'BOS O O\n'
      for word, tag, tag_pred in zip(words, tags, pred):
        out += '{} {} {}\n'.format(word, tag, tag_pred)

      out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1]) # remove the ending \n on last line
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''

    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    output = subprocess.getoutput("perl conlleval.pl < {}".format(filename))

    for line in output.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    print({'p': precision, 'r': recall, 'f1': f1score})
    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the word sequence.
    target_path: path to the file with token-ids for the tag sequence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    label_path: path to the file with token-ids for the intent label
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target, label) tuple read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1];source, target, label are lists of token-ids
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(label_path, mode="r") as label_file:
        source = source_file.readline()
        target = target_file.readline()
        label = label_file.readline()
        counter = 0
        while source and target and label and (not max_size \
                                               or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            logging.info('\treading data line %s', counter)
            sys.stdout.flush()

          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          label_ids = [int(x) for x in label.split()]
#          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, label_ids])
              break
          source = source_file.readline()
          target = target_file.readline()
          label = label_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

def create_model(session, 
                 source_vocab_size, 
                 target_vocab_size, 
                 label_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          label_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=False, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task, 
          embedding_path=FLAGS.embedding_path)
  with tf.variable_scope("model", reuse=True):
    model_test = multi_task_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          label_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, 
          FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=True, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task, 
          embedding_path=FLAGS.embedding_path)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    logging.info('Reading model parameters from %s', ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    logging.info('Created model with fresh parameters.')
    session.run(tf.global_variables_initializer())
  return model_train, model_test
        
def train():
  logging.info('Applying Parameters:')
  logging.info('Preparing data in %s', FLAGS.data_dir)

  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''
  date_set = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
  in_seq_train, out_seq_train, label_train = date_set[0]
  in_seq_dev, out_seq_dev, label_dev = date_set[1]
  in_seq_test, out_seq_test, label_test = date_set[2]
  vocab_path, tag_vocab_path, label_vocab_path = date_set[3]
     
  result_dir = FLAGS.train_dir + '/test_results'
  if not os.path.isdir(result_dir):
      os.makedirs(result_dir)

  current_intent_valid_out_file = result_dir + '/intent.valid.hyp.txt'
  current_intent_test_out_file = result_dir + '/intent.test.hyp.txt'

  current_taging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
  current_taging_test_out_file = result_dir + '/tagging.test.hyp.txt'

  vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

  config = tf.ConfigProto(
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
      #device_count = {'gpu': 2}
  )
    
  with tf.Session(config=config) as sess:
    # Create model.
    logging.info('Max sequence length: %s.', _buckets[0][0])
    logging.info('Creating %s layers of %s units.', FLAGS.num_layers, FLAGS.size)
    
    model, model_test = create_model(sess, 
                                     len(vocab), 
                                     len(tag_vocab), 
                                     len(label_vocab))
    logging.info(
      'Creating model with source_vocab_size=%s, target_vocab_size=%s, label_vocab_size=%s.', 
      len(vocab), len(tag_vocab), len(label_vocab)
    )

    # Read data into buckets and compute their sizes.
    logging.info('Reading train/valid/test data (training set limit: %s).', FLAGS.max_train_data_size)
    dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
    test_set = read_data(in_seq_test, out_seq_test, label_test)
    train_set = read_data(in_seq_train, out_seq_train, label_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0

    best_valid_score = 0
    best_test_score = 0
    while model.global_step.eval() < FLAGS.max_training_steps:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      batch_data = model.get_batch(train_set, bucket_id)
      encoder_inputs,tags,tag_weights,batch_sequence_length,labels = batch_data
      if task['joint'] == 1:
        step_outputs = model.joint_step(sess, 
                                        encoder_inputs, 
                                        tags, 
                                        tag_weights, 
                                        labels, 
                                        batch_sequence_length, 
                                        bucket_id, 
                                        False)
        _, step_loss, tagging_logits, class_logits = step_outputs
      elif task['tagging'] == 1:
        step_outputs = model.tagging_step(sess, 
                                          encoder_inputs,
                                          tags,
                                          tag_weights,
                                          batch_sequence_length, 
                                          bucket_id, 
                                          False)
        _, step_loss, tagging_logits = step_outputs
      elif task['intent'] == 1:
        step_outputs = model.classification_step(sess, 
                                                 encoder_inputs, 
                                                 labels,
                                                 batch_sequence_length, 
                                                 bucket_id, 
                                                 False)  
        _, step_loss, class_logits = step_outputs

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d step-time %.2f. Training perplexity %.2f" 
            % (model.global_step.eval(), step_time, perplexity))
        sys.stdout.flush()
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0 
        
        def run_valid_test(data_set, mode): # mode: Eval, Test
            """ Run evals on development/test set and print the accuracy """
            word_list = list() 
            ref_tag_list = list() 
            hyp_tag_list = list()
            ref_label_list = list()
            hyp_label_list = list()
            correct_count = 0
            accuracy = 0.0
            tagging_eval_result = dict()
            for bucket_id in xrange(len(_buckets)):
              eval_loss = 0.0
              count = 0
              for i in xrange(len(data_set[bucket_id])):
                count += 1
                sample = model_test.get_one(data_set, bucket_id, i)
                encoder_inputs,tags,tag_weights,sequence_length,labels = sample
                tagging_logits = []
                class_logits = []
                if task['joint'] == 1:
                  step_outputs = model_test.joint_step(sess, 
                                                       encoder_inputs, 
                                                       tags, 
                                                       tag_weights, 
                                                       labels,
                                                       sequence_length, 
                                                       bucket_id, 
                                                       True)
                  _, step_loss, tagging_logits, class_logits = step_outputs
                elif task['tagging'] == 1:
                  step_outputs = model_test.tagging_step(sess, 
                                                         encoder_inputs, 
                                                         tags, 
                                                         tag_weights,
                                                         sequence_length, 
                                                         bucket_id, 
                                                         True)
                  _, step_loss, tagging_logits = step_outputs
                elif task['intent'] == 1:
                  step_outputs = model_test.classification_step(sess, 
                                                                encoder_inputs, 
                                                                labels,
                                                                sequence_length, 
                                                                bucket_id, 
                                                                True) 
                  _, step_loss, class_logits = step_outputs
                eval_loss += step_loss / len(data_set[bucket_id])
                hyp_label = None
                # Predict intent
                if task['intent'] == 1:
                  ref_label_list.append(rev_label_vocab[labels[0][0]])
                  hyp_label = np.argmax(class_logits[0], 0)
                  hyp_label_list.append(rev_label_vocab[hyp_label])
                  if labels[0] == hyp_label:
                    correct_count += 1

                # Predict slots
                if task['tagging'] == 1:
                  word_list.append([rev_vocab[x[0]] for x in \
                                    encoder_inputs[:sequence_length[0]]])
                  ref_tag_list.append([rev_tag_vocab[x[0]] for x in \
                                       tags[:sequence_length[0]]])
                  hyp_tag_list.append(
                          [rev_tag_vocab[np.argmax(x)] for x in \
                                         tagging_logits[:sequence_length[0]]])

            accuracy = float(correct_count)*100/count
            if task['intent'] == 1:
              if mode == 'Eval':
                  intent_out_file = current_intent_valid_out_file
              elif mode == 'Test':
                  intent_out_file = current_intent_test_out_file

              print("  %s accuracy: %.2f %d/%d" \
                    % (mode, accuracy, correct_count, count))
              sys.stdout.flush()
              intenteval(hyp_label_list, intent_out_file, mode)
            if task['tagging'] == 1:
              if mode == 'Eval':
                  taging_out_file = current_taging_valid_out_file
              elif mode == 'Test':
                  taging_out_file = current_taging_test_out_file
              
              tagging_eval_result = conlleval(
                hyp_tag_list, 
                ref_tag_list, 
                word_list, 
                taging_out_file,
                mode
              )
              print("  %s f1-score: %.2f" % (mode, tagging_eval_result['f1']))
              sys.stdout.flush()
              #sys.exit(0)
            return accuracy, tagging_eval_result
            
        # valid
        valid_accuracy, valid_tagging_result = run_valid_test(dev_set, 'Eval')        
        if task['tagging'] == 1 \
            and valid_tagging_result['f1'] > best_valid_score:
          best_valid_score = valid_tagging_result['f1']
          # save the best output file
          subprocess.call(['mv', 
                           current_taging_valid_out_file, 
                           current_taging_valid_out_file + '.best_f1_%.2f' \
                           % best_valid_score])
          subprocess.call(['mv', 
                           current_intent_valid_out_file, 
                           current_intent_valid_out_file + '.best_f1_%.2f' \
                           % best_valid_score])
        # test, run test after each validation for development purpose.
        test_accuracy, test_tagging_result = run_valid_test(test_set, 'Test')        
        if task['tagging'] == 1 \
            and test_tagging_result['f1'] > best_test_score:
          best_test_score = test_tagging_result['f1']
          # save the best output file
          subprocess.call(['mv', 
                           current_taging_test_out_file, 
                           current_taging_test_out_file + '.best_f1_%.2f' \
                           % best_test_score])
          subprocess.call(['mv', 
                           current_intent_test_out_file, 
                           current_intent_test_out_file + '.best_f1_%.2f' \
                           % best_test_score])
                    
          
def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
