import tensorflow as tf
from NetworkBuilder import NetworkBuilder
from DatasetGenerater import DataSetGenerator
import datetime
import os
import  numpy as np

with tf.name_scope('Input_layer') as scope:
    Input = tf.placeholder(dtype='float',shape=[None,128,128,1], name="Input")

with tf.name_scope("Target_layer") as scope:
    Target = tf.placeholder(dtype='float',shape=[None,2], name="Target")


nb=NetworkBuilder()

with tf.name_scope("MyAwesomeModel") as scope:
    model=Input;
    model=nb.attach_conv_layer(model,32,summary=True)
    model= nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_flatten_layer(model)
    model = nb.attach_dense_layer(model,200,summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 30, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 2, summary=True)
    model = nb.attach_softmax_layer(model)
    prediction=model

with tf.name_scope('Optimizer') as scope:
    global_itr=tf.Variable(0,name='global_itr',trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Target, name='softmax_cost_function')
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost",cost)

    optmizer=tf.train.AdamOptimizer().minimize(cost, global_step=global_itr);

with tf.name_scope('accuracy') as scope:
    accu=tf.equal( tf.arg_max(prediction,1),tf.arg_max(Target,1))
    accu=tf.reduce_mean(tf.cast(accu,'float'))


dg=DataSetGenerator("train")

saver=tf.train.Saver()
model_save_path='./my saved model/'
modelname='myFirstModel'


epochs=10
batchSize=10

with tf.Session() as sess:
    summaryMerged=tf.summary.merge_all()
    filePath="./summary_log/run"+datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    writer = tf.summary.FileWriter(filePath, sess.graph)

    tf.global_variables_initializer().run()

    if(os.path.exists(model_save_path+"checkpoint")):
        saver.restore(sess,model_save_path)

    for epoch in range(epochs):
        batches=dg.get_mini_batches(batchSize,(128,128),allchannel=False)
        for imgs ,labels in batches:
            imgs= np.divide(imgs,255)
            err,accuracy, summ, i, _ = sess.run([cost, accu, summaryMerged, global_itr, optmizer],
                                                feed_dict={Input:imgs,Target:labels})

            writer.add_summary(summ,i)
            print("epoches:",epoch," err=",err," accu=",accuracy)
            if(i%100)==0:
               print("saving Model")
               saver.save(sess,model_save_path+modelname)
