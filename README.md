模型训练分布在4个节点进行测试yanzheng12：

    一台ps服务器，一台chief服务器（master节点），两台worker服务器
Train_model_DNN_Cluster.py,Train_model_Linear_Cluster模型训练代码中指定了
cluster集群的配置：

    cluster = {'chief': ['tf-2:22221'],
                             'ps': ['tf-1:22222'],
                              'worker': ['tf-3:22223','tf-4:22224']}
                              
    cluster = tf.train.ClusterSpec({'chief': ['tf-2:22221'],
                        'ps': ['tf-1:22222'],
                         'worker': ['tf-3:22223','tf-4:22224']}) 
                         
    根据不同运行环境修改配置。 
                            
提交命令：ps节点

        （tf-1节点）python ./Train_model_DNN_Cluster.py --job_name ps --task_index 0
      chief节点
        （tf-2节点）python ./Train_model_DNN_Cluster.py --job_name chief --task_index 0
      worker节点
        （tf-3节点）python ./Train_model_DNN_Cluster.py --job_name worker --task_index 0
        （tf-4节点）python ./Train_model_DNN_Cluster.py --job_name worker --task_index 1
pred_Linear.py,pred_DNN.py模型预测:

    加载Linear_model_file里的模型数据时，要把tf-1与tf-2节点的模型数据整合到一个文件里
Loss_weight:

    DNN_test.py,Linear_test.py对模型准确度的计算（MSE,MAE)
    Get_Weight.py获取模型特征值 
详细的模型训练方法以及预测请查看runs.ipynb        
