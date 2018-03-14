# -*- coding:utf-8 -*- 
'''wide and deep模型的核心思想是结合线性模型的记忆能力（memorization）和
DNN模型的泛化能力（generalization），在训练过程中同时优化2个模型的参数，
从而达到整体模型的预测能力最优
DNN：深度神经网络在做有监督学习前要先做非监督学习，然后将非监督学习学到的权值当作有监督学习的初值进行训练'''

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='DNN_model_file',
    help='Base directory for the DNN model.')

parser.add_argument(
    '--train_epochs', type=int, default=250, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=100, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='data/20180128.csv',
    help='Path to the training data.')
# 设置job name参数
parser.add_argument('--job_name', type=str,default = None, 
                    help='job name: worker or ps')
# 设置任务的索引
parser.add_argument('--task_index',  type=int,default =None,
                    help= 'Index of task within the job') 

# build features
def build_model_columns():
    x1 = tf.feature_column.numeric_column('x1')
    x2 = tf.feature_column.numeric_column('x2')
    x3 = tf.feature_column.numeric_column('x3')
    x4 = tf.feature_column.numeric_column('x4')
    x5 = tf.feature_column.numeric_column('x5')
    # 存储分区化列
    x1_buckets = tf.feature_column.bucketized_column(
        x1, boundaries=np.linspace(0.21,0.41,num=21).tolist())
    x2_buckets = tf.feature_column.bucketized_column(
        x2, boundaries=np.linspace(20,25,num=6).tolist())
    x3_buckets = tf.feature_column.bucketized_column(
        x3, boundaries=[2,2.8])
    x4_buckets = tf.feature_column.bucketized_column(
        x4, boundaries=np.linspace(20,200,num=10).tolist())
    x5_buckets = tf.feature_column.bucketized_column(
        x5, boundaries=np.linspace(20,100,num=11).tolist())

    base_columns = [x1,x2,x3,x4,x5]
    # 广度模型：具有交叉特征列的线性模型
    crossed_columns = [
        tf.feature_column.crossed_column([x1_buckets,x2_buckets], hash_bucket_size=1000),
        tf.feature_column.crossed_column([x3_buckets,x4_buckets,x5_buckets], hash_bucket_size=1000)
    ]

    wide_columns = base_columns + crossed_columns
    # 深度模型：嵌入式神经网络
    # 嵌入的 dimension 越高，自由度就越高，模型将不得不学习这些特性的表示。特征列的维度为 10（类型特征）
    deep_columns = [
        #         tf.feature_column.indicator_column(x1_buckets),
        #         tf.feature_column.indicator_column(x2_buckets),
        #         tf.feature_column.indicator_column(x3_buckets),
        #         tf.feature_column.indicator_column(x4_buckets),
        #         tf.feature_column.indicator_column(x5_buckets),
        tf.feature_column.embedding_column(x1_buckets, dimension=10),
        tf.feature_column.embedding_column(x2_buckets, dimension=10),
        tf.feature_column.embedding_column(x3_buckets, dimension=10),
        tf.feature_column.embedding_column(x4_buckets, dimension=10),
        tf.feature_column.embedding_column(x5_buckets, dimension=10)
    ]
    return wide_columns, deep_columns

def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [150, 75, 50]
    # 添加TF_CONFIG变量，及集群配置
    cluster = {'chief': ['tf-2:22221'],
                'ps': ['tf-1:22222'],
                 'worker': ['tf-3:22223','tf-4:22224']}
    os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
# 将广度和深度模型结合为一体
# 具有稀疏特征列和大量可能特征值的数据集，广度和深度学习将会更加强大
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
#         输入线性模型的feature columns  线性模型的输入特征
        linear_feature_columns=wide_columns,
#         输入DNN模型的feature columns DNN模型的输入特征
        dnn_feature_columns=deep_columns,
#       dnn_hidden_units  每个隐藏层的神经元数目
        dnn_hidden_units=hidden_units,
#         dnn_activation_fn 隐藏层的激活函数，默认采用RELU
        dnn_activation_fn=tf.nn.relu,
#         模型训练中隐藏层单元的drop_out比例
        #dnn_dropout = 0.5,
        config=run_config)

def main(unused_argv):
    # Clean up the model directory if present
    df = pd.read_csv(FLAGS.train_data)
    
    x_train_dict = {'x1':df['x1'].values,
                    'x2':df['x2'].values,
                    'x3':df['x3'].values,
                    'x4':df['x4'].values,
                    'x5':df['x5'].values }
    y_train = df['y'].values
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    #     input_fn用于特征和目标数据传递给train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = x_train_dict,
        y = y_train,
        batch_size = FLAGS.batch_size,
        num_epochs = FLAGS.train_epochs,
        shuffle=True)

    cluster = tf.train.ClusterSpec({'chief': ['tf-2:22221'],
                'ps': ['tf-1:22222'],
                 'worker': ['tf-3:22223','tf-4:22224']})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
