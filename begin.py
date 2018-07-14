import tensorflow as tf
import pandas as pd
import numpy as np

data=pd.read_csv('iris.csv',names=['c1','c2','c3','c4','c5'])
features=data[[['c1','c2','c3','c4']]]
label=data['c5']

a=np.asarray([1,0,0])
b=np.asarray([0,1,0])
c=np.asarray([0,0,1])

data['c5']=data['c5'].map({'Iris-setosa':a,'Iris-versicolor':b,'Iris-virginica':c})
label=data['c5']
j=np.random.permutation(len(data))
features=features.iloc[j]
label=data['c5'].iloc[j]

features=features.reset_index(drop=True)
label=label.reset_index(drop=True)

train_features=features[0:106]
train_labels=label[0:106]
test_features=features[106:]
test_labels=label[106:]

x=tf.placeholder(tf.float32,[None,4])
y=tf.placeholder(tf.float32,[None,3])

W1=tf.Variable(tf.random_normal([4,10]),tf.float32)
b1=tf.Variable(tf.random_normal([10]),tf.float32)
y1=tf.nn.sigmoid(tf.matmul(x,W1)+b1)

W2=tf.Variable(tf.random_normal([10,20]),tf.float32)
b2=tf.Variable(tf.random_normal([20]),tf.float32)
y2=tf.nn.sigmoid(tf.matmul(y1,W2)+b2)

W3=tf.Variable(tf.random_normal([20,3]),tf.float32)
b3=tf.Variable(tf.random_normal([3]),tf.float32)
yhat=tf.nn.softmax(tf.matmul(y2,W3)+b3)

cost=tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))

optimize=tf.train.GradientDescentOptimizer(0.5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        
        sess.run(optimize,feed_dict={x:train_features,y:[t for t in train_labels.as_matrix()]})
        sess.run(W3)
    correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: train_features, y:[t for t in train_labels.as_matrix()]}))
    
    
    
    




