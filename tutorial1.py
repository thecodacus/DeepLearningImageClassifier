import tensorflow as tf


Target = tf.placeholder('float', [None, 1], name='Target')

with tf.name_scope("Input_Layer") as scope:
    Input = tf.placeholder('float', [None, 2], name='Input')
    inputBias = tf.Variable(initial_value=tf.random_normal(shape=[3], stddev=0.4), dtype='float', name="input_bias")

with tf.name_scope("Hidden_Layer") as scope:
    weights = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4), dtype='float', name="hidden_weights")
    hiddenBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4), dtype='float', name="hidden_bias")
    tf.summary.histogram(name="Weights_1", values=weights)
    hiddenLayer = tf.matmul ( Input , weights ) + inputBias
    hiddenLayer = tf.sigmoid ( hiddenLayer , name='hidden_layer_activation' )



with tf.name_scope("Output_layer") as scope:
    outputWeights = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4), dtype='float',name="output_weights")
    tf.summary.histogram(name="Weights_2", values=outputWeights)
    output = tf.matmul(hiddenLayer, outputWeights)+hiddenBias
    output = tf.sigmoid(output, name='output_layer_activation')


with tf.name_scope("Optimiser") as scope:
    cost = tf.squared_difference(Target, output)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("error_1", cost)
    tf.summary.scalar("error_2", cost)
    tf.summary.scalar("lr_1", cost)
    tf.summary.scalar("lr_2", cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

inp = [[0, 0], [0, 1], [1, 0], [1, 1]]
out = [[0], [1], [1], [0]]

epochs = 4000  # number of time we want to repeat
import datetime
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mergedSummary = tf.summary.merge_all()
    fileName = "./summary_log/run"+datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
    writer=tf.summary.FileWriter(fileName,sess.graph)
    for i in range(epochs):
        err, _,summaryOutput = sess.run([cost, optimizer, mergedSummary], feed_dict={Input: inp, Target: out})
        writer.add_summary(summaryOutput,i)

    while True:
        inp = [[0, 0]]
        inp[0][0] = input("type 1st input :")
        inp[0][1] = input("type 2nd input :")
        print(sess.run([output], feed_dict={Input: inp})[0][0])
