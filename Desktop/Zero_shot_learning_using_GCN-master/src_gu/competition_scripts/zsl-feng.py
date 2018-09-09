from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn import preprocessing
import cv2

from config_lan import FLAGS


def load_Img(imgDir, image_size=256):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    imgNum = len(imgs)
    data = np.empty((imgNum,image_size,image_size,3),dtype="float32")
    for i in range (imgNum):
        img = Image.open(imgDir+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        arr = cv2.resize(arr,(image_size,image_size))
        if len(arr.shape) == 2:     # some images just has 2 channels
            temp = np.empty((image_size,image_size,3))
            temp[:,:,0] = arr
            temp[:,:,1] = arr
            temp[:,:,2] = arr
            arr = temp
        data[i,:,:,:] = arr
    return data


def make_label(labelFile):
    label_list = pd.read_csv(labelFile,sep = '\t',header = None)
    label_list = label_list.sort_values(by=0)
    le = preprocessing.LabelEncoder()
    for item in [1]:
        label_list[item] = le.fit_transform(label_list[item])
    label = label_list[1].values
    onehot = preprocessing.OneHotEncoder(sparse = False)
    label_onehot = onehot.fit_transform(np.mat(label).T)
    return label_onehot


def make_attributes_label(attributes_per_classFile):
    attributes_per_class = pd.read_csv(attributes_per_classFile,sep='\t',header=None)
    label_list = pd.read_csv(labelFile,sep = '\t',header = None)
    table = label_list.set_index(1).join(attributes_per_class.set_index(0)).sort_values(by=0).reset_index()
    del table['index']
    del table[0]
    return table.values


def data_slice(data):
    loc_x = int(32*np.random.random())
    loc_y = int(32*np.random.random())
    return data[:,loc_x:loc_x+224,loc_y:loc_y+224,:]


def testdata_slice(data, flag):
    if flag == 0:
        return data[:,0:224,0:224,:]
    elif flag == 1:
        return data[:,0:224,-1-224:-1,:]
    elif flag == 2:
        return data[:,-1-224:-1,0:224,:]
    elif flag == 3:
        return data[:,-1-224:-1,-1-224:-1,:]
    elif flag == 4:
        return data[:,16:240,16:240,:]


batch_size = 50
num_batches = 300000
image_size = 224

imgDir = FLAGS.img_dir
labelFile = FLAGS.train_file
attributes_per_classFile = FLAGS.attrs_per_class_dir

traindata = load_Img(imgDir)
traindata = traindata/255.
trainlabel = make_label(labelFile)
train_attributes_label = make_attributes_label(attributes_per_classFile)
print(traindata.shape,trainlabel.shape,train_attributes_label.shape)

test_imgDir = FLAGS.test_img_dir
testdata = load_Img(test_imgDir)


with tf.Graph().as_default():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    my_graph = tf.Graph()
    sess = tf.InteractiveSession(graph=my_graph,config=config)

    x = tf.placeholder(tf.float32,[None,image_size*image_size*3])
    x = tf.reshape(x,[-1,image_size,image_size,3])
    y = tf.placeholder(tf.float32,[None,train_attributes_label.shape[-1]])

    y_conv,_ = resnet_v2.resnet_v2_152(x,num_classes=train_attributes_label.shape[-1])
    y_conv = tf.reshape(y_conv,[-1,30])
    y_conv = tf.nn.sigmoid(y_conv)

    cross_entropy = tf.reduce_mean(tf.square(y_conv-y))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

    tf.summary.scalar('loss',cross_entropy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./", sess.graph)


    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    i = 0
    rand_index = np.random.choice(35221,size=(100))
    while (cross_entropy.eval(feed_dict={x:data_slice(traindata[rand_index]),y:train_attributes_label[rand_index]}) >=0.001) and i <= num_batches:
        i += 1
        rand_index = np.random.choice(35221,size=(batch_size))
        print(traindata[rand_index].shape)
        train_step.run(feed_dict={x:data_slice(traindata[rand_index]),y:train_attributes_label[rand_index]})
        if i%50 == 0:
            rand_index = np.random.choice(35221,size=(100))
            ceshi_data = data_slice(traindata[rand_index])
            ceshi_label = train_attributes_label[rand_index]
            rs = sess.run(merged,feed_dict={x:ceshi_data,y:ceshi_label})
            writer.add_summary(rs,i)
            print('step %d, the cross entropy is %g'%(i,cross_entropy.eval(feed_dict={x:ceshi_data,y:ceshi_label})))
            print(y_conv.eval(feed_dict={x:ceshi_data,y:ceshi_label}))
            print(y.eval(feed_dict={x:ceshi_data,y:ceshi_label}))

    prelist_lis = []
    for flag in range(5):
        pre_list = []
        for i in range(147):
            if i == 146:
                index = np.arange(14600,14633,1)
            else:
                index = np.arange(i*100,(i+1)*100,1)
            l = y_conv.eval(feed_dict={x:testdata_slice(testdata[index],flag),y:train_attributes_label[index]})
            pre_list += [l]
        res = pre_list[0]
        for i in list(np.arange(1,147,1)):
            res = np.row_stack([res,pre_list[i]])
        print(res.shape)
        prelist_lis.append(res)
    writer.close()
    saver.save(sess,"./test1_inception_v3.ckpt")
    sess.close()


label_attributes = pd.read_csv(attributes_per_classFile,sep='\t',header=None)
label_attributes = label_attributes.set_index(0)


labellist_lis = []
loclist_lis = []
for j in range(5):
    res = prelist_lis[j]
    label_lis = []
    loc_lis = []
    for i in range(res.shape[0]):
        pre_res = res[i,:]
        loc = np.sum(np.abs(label_attributes.values - pre_res),axis=1).argmin()
        label_lis.append(label_attributes.index[loc])
        loc_lis.append(loc)
    labellist_lis.append(label_lis)
    loclist_lis.append(loc_lis)

label1 = labellist_lis[0]
label2 = labellist_lis[1]
label3 = labellist_lis[2]
label4 = labellist_lis[3]
label5 = labellist_lis[4]

loc1 = loclist_lis[0]
loc2 = loclist_lis[1]
loc3 = loclist_lis[2]
loc4 = loclist_lis[3]
loc5 = loclist_lis[4]

res_all = []
for i in range(len(loc1)):
    vote = [0]*230
    vote[loc1[i]] += 1
    vote[loc2[i]] += 1
    vote[loc3[i]] += 1
    vote[loc4[i]] += 1
    vote[loc5[i]] += 1
    if max(vote) == 1:
        res_all.append(label_attributes.index[loc5[i]])
    else:
        res_all.append(label_attributes.index[vote.index(max(vote))])

imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = res_all
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_resall.txt',sep = '\t',header = None,index = None)


imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = label1
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_res1.txt',sep = '\t',header = None,index = None)

imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = label2
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_res2.txt',sep = '\t',header = None,index = None)

imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = label3
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_res3.txt',sep = '\t',header = None,index = None)

imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = label4
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_res4.txt',sep = '\t',header = None,index = None)

imgs = os.listdir(test_imgDir)
imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
testlabel = pd.DataFrame(imgs)
testlabel['label'] = label5
print(testlabel)
testlabel.to_csv('./res_2018_8_24_attr_res5.txt',sep = '\t',header = None,index = None)
