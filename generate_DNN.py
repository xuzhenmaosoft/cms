import argparse
import shutil
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='DNN_model_file',
    help='Base directory for the DNN model.')

parser.add_argument(
    '--precision', type=str, default='low',
    help='precision used to generate the lookup table')

parser.add_argument(
    '--test_dir', type=str, default='result',
    help='Path to the result data.')

#build lookup table

def build_table(precision):
    
    if precision == 'high':
        x1 = np.linspace(0.21,0.42,num=22)  
        x2 = np.linspace(19,26,num=71)  
        x3 = [2.5,3]  
        x4 = np.linspace(0,176,num=177)  
        x5 = np.linspace(3,106,num=104)
    
    elif precision =='low':
        x1 = np.linspace(0.21,0.42,num=22)  
        x2 = np.linspace(19,26,num=71)  
        x3 = [2.5,3]  
        x4 = np.linspace(94,176,num=83)  
        x5 = np.linspace(51,106,num=56) 
        
    elif precision =='demo':
        x1 = np.linspace(0.21,0.42,num=22)  
        x2 = np.linspace(19,26,num=71)  
        x3 = [2.5,3]  
        x4 = np.linspace(156,176,num=21)  
        x5 = np.linspace(100,106,num=7)         

    else:
        raise ValueError('please set precision eitehr low or high')
        
    res = np.array(np.meshgrid(x1,x2,x3,x4,x5)).T.reshape(-1,5)
    return pd.DataFrame(res,columns=['x1','x2','x3','x4','x5'])

# build features
def build_model_columns():
    x1 = tf.feature_column.numeric_column('x1')
    x2 = tf.feature_column.numeric_column('x2')
    x3 = tf.feature_column.numeric_column('x3')
    x4 = tf.feature_column.numeric_column('x4')
    x5 = tf.feature_column.numeric_column('x5')

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

    crossed_columns = [
        tf.feature_column.crossed_column([x1_buckets,x2_buckets], hash_bucket_size=1000),
        tf.feature_column.crossed_column([x3_buckets,x4_buckets,x5_buckets], hash_bucket_size=1000)
    ]

    wide_columns = base_columns + crossed_columns

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

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        dnn_activation_fn=tf.nn.relu,
        #dnn_dropout = 0.5,
        config=run_config)

def main(unused_argv):
    # Clean up the model directory if present
    df = build_table(FLAGS.precision)
    
    x_test_dict = {'x1':df['x1'].values,
                    'x2':df['x2'].values,
                    'x3':df['x3'].values,
                    'x4':df['x4'].values,
                    'x5':df['x5'].values }
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = x_test_dict,
        y = None,
        batch_size = df.shape[0],
        num_epochs = 1,
        shuffle=False)
    
    #shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir)

    res_pred = model.predict(input_fn=test_input_fn)
    preds= []
    for pred in res_pred:
        preds.append(pred['predictions'][0])
    df['y_pred'] = np.array(preds)
    
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)
    df.to_csv("{}/generated_pred_dnn.csv".format(FLAGS.test_dir),index=False)
    print('predicitons saved at {}/generated_pred_dnn.csv'.format(FLAGS.test_dir))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
