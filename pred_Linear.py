import argparse
import shutil
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='Linear_model_file',
    help='Base directory for the Linear model.')

parser.add_argument(
    '--test_data', type=str, default='data/20180128.csv',
    help='Path to the test data.')

parser.add_argument(
    '--test_dir', type=str, default='result',
    help='Path to the result data.')

# build features
def build_model_columns():
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
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    return tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001),
        config=run_config)

def main(unused_argv):
    # Clean up the model directory if present
    df = pd.read_csv(FLAGS.test_data)
    
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
    df_preds = pd.DataFrame(preds,columns=['y_pred'])
    
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)
    df_preds.to_csv("{}/pred_linear.csv".format(FLAGS.test_dir),index=False)
    print('predicitons saved at {}/pred_linear.csv'.format(FLAGS.test_dir))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
