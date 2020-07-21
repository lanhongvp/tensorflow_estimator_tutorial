# tensorflow_estimator_tutorial
- The tensorflow version is out of date, please pay attention to the version problem. 
**Enjoy tf.estimator**

## 代码结构
```
|--tensorflow_estimator_learn               
    |--data_csv
        |--mnist_test.csv
        |--mnist_train.csv
        |--mnist_val.csv
        
    |--images
        |--ZJUAI_2018_AUT
        |--ZJUAI_2018_AUT

    |--tmp
        |--ZJUAI_2018_AUT
        |--ZJUAI_2018_AUT
        
    |--CNNClassifier.jpynb
    
    |--CNNClassifier_dataset.jpynb    
    
    |--CNN_raw.jpynb    
    
    |--DNNClassifier.jpynb    
    
    |--DNNClassifier_dataset.jpynb    
```
## 文件说明
### data_csv
data_csv文件中存放了**MNSIT**原始csv文件，分为验证、训练、测试三个部分
### images
images文件中存放了**jupyter notebook**中所涉及的一些图片
### tmp
tmp 文件中存放了一些临时代码
### CNNClassifier.jpynb
未采用`tf.data`API的自定义estimator实现
### CNNClassifier_dataset.jpynb
采用`tf.data`API的自定义estimator实现
### CNN_raw.jpynb
未采用高阶API的 **搭建CNN实现MNIST分类**
### DNNClassifier.jpynb 
未采用`tf.data`API的预制sestimator实现
### DNNClassifier_dataset.jpynb
采用`tf.data`API的预制estimator实现

