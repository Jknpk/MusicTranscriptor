from random import shuffle
import glob
import os
import sys
import numpy as np 

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2  #sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    #print(img[0,153])
    #img = img.astype(np.float32)
    #print(img.type())
    #print(addr)
    #print(img.shape)
    #print(img.dtype)
    #print(img[50,50])
    return img


def storeTrainData(train_addrs, train_labels):
    train_filename = 'train.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
        if not i % 10:
            print('Train data: {}/{}'.format(i, len(train_addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        label = train_labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def generateTrainData(path = ('../trainSet1/constQSpectrograms/*.png')):
    shuffle_data = True  # shuffle the addresses before saving

    addrs = []
    labels = []

    for spectrograms_train_path in path:
        print(spectrograms_train_path)
        # read addresses and labels from the 'train' folder
        addrs.extend(glob.glob(spectrograms_train_path))
        labels = []
        for addr in addrs:
            baseName = os.path.basename(addr)
            pureName = os.path.splitext(baseName)[0]
            #print(str)

            numbersInPureName = [int(s) for s in pureName.split('_') if s.isdigit()]
            label = numbersInPureName[0]
            labels.append(label)
            #print(label)

        #print(labels)
        #print("\n") 
        #print(addrs)


    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)


    storeTrainData(addrs, labels)



def addTestData(test_addrs, test_labels):
    test_filename = 'test.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
        if not i % 10:
            print('Test data: {}/{}'.format(i, len(test_addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(test_addrs[i])
        label = test_labels[i]
        # Create a feature
        feature = {'test/label': _int64_feature(label),
                   'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def generateTestData(path='../trainSet1/constQSpectrograms/*.png'):
    shuffle_data = False  # shuffle the addresses before saving
    spectrograms_test_path = path
    # read addresses and labels from the 'train' folder
    addrs = glob.glob(spectrograms_test_path)
    labels = []
    for addr in addrs:
        baseName = os.path.basename(addr)
        pureName = os.path.splitext(baseName)[0]
        #print(str)

        numbersInPureName = [int(s) for s in pureName.split('_') if s.isdigit()]
        label = numbersInPureName[0]
        labels.append(label)
        #print(label)

    #print(labels)
    #print("\n") 
    #print(addrs)


    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)


    addTestData(addrs, labels)



generateTrainData(('../trainSet5/constQSpectrograms/*.png', '../trainSet6/constQSpectrograms/*.png', '../trainSet7/constQSpectrograms/*.png', '../trainSet8/constQSpectrograms/*.png', '../trainSet9/constQSpectrograms/*.png'))#, '../trainSet6/pseudoQSpectrograms/*.png'))
generateTestData('../trainSet10/constQSpectrograms/*.png')

#generateTrainData()




#print("\n")

#print(labels)
#print("\n") 
#print(addrs)


#img = load_image(addrs[0])
#cv2.imshow("image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
   

   
# Divide the data into 60% train, 20% validation, and 20% test

'''
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]
'''
