# -*- coding:utf-8 -*- 
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


#argpres 解析命令行参数和选项的标准模块
parser = argparse.ArgumentParser()
#add_argument 对应一个你要关注的参数或选项
parser.add_argument(
    '--model_dir', type=str, default='Linear_model_file',
    help='Base directory for the Linear model.')

parser.add_argument(
    '--train_epochs', type=int, default=200, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=100, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='data/20180128.csv',
    help='Path to the training data.')
parser.add_argument(
    '--test_dir', type=str, default='result',
    help='Path to the result data.')
# 设置job name参数
parser.add_argument('--job_name', type=str,default = None, 
                    help='job name: worker or ps')
# 设置任务的索引
parser.add_argument('--task_index',  type=int,default =None,
                    help= 'Index of task within the job')

# build features
def build_model_columns():
#     feature_column.numeric_column 建立数值 特征列
    x1 = tf.feature_column.numeric_column('x1')
    x2 = tf.feature_column.numeric_column('x2')
    x3 = tf.feature_column.numeric_column('x3')
    x4 = tf.feature_column.numeric_column('x4')
    x5 = tf.feature_column.numeric_column('x5')

    wide_columns = [x1,x2,x3,x4,x5]

    return wide_columns

def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    wide_columns = build_model_columns()

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
# 设置TF_CONFIG变量，及集群配置信息

    cluster = {'chief': ['tf-2:22221'],
                'ps': ['tf-1:22222'],
                 'worker': ['tf-3:22223','tf-4:22224']}
    os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
#     创建评估器
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'CPU': 0}))
# 给Estimator提供了一个函数model_fn(LinearRegressor)，它告诉tf.estimator如何评估预测，训练步骤和损失
# model_dir 模型存放位置

    return tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        
        config=run_config)

def main(unused_argv):
    # Clean up the model directory if present
    df = pd.read_csv(FLAGS.train_data)
#     构建训练集
    x_train_dict = {'x1':df['x1'].values,
                    'x2':df['x2'].values,
                    'x3':df['x3'].values,
                    'x4':df['x4'].values,
                    'x5':df['x5'].values }
    y_train = df['y'].values
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = x_train_dict,
        y = y_train,
#         每一轮训练（计算一次损失函数）的数据量大小
        batch_size = FLAGS.batch_size,
#         每一个迭代训练多少批次/训练过程中数据将被“轮”多少次
        num_epochs = FLAGS.train_epochs,
        shuffle=True)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({'chief': ['tf-2:22221'],
                'ps': ['tf-1:22222'],
                 'worker': ['tf-3:22223','tf-4:22224']})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
#     input_fn用于特征和目标数据传递给train
    else:
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:%s/task:%d/CPU:0" % (FLAGS.job_name , FLAGS.task_index),
            cluster=cluster)):
            shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
            model = build_estimator(FLAGS.model_dir)

        # tf.train.Scaffold(saver=FLAGS.model_dir)
        sv = tf.train.MonitoredTrainingSession(is_chief=(FLAGS.job_name == "chief")
                                 )
        with sv as sess:
            model.train(input_fn=train_input_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
#     parse_known_args 解析命令行参数
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
