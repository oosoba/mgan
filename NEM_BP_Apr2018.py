## OA: Apr-2018
import numpy as np
import tensorflow as tf
import math

# Unpack the dataset
def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict
	
# Unpacking training and test data
b1 = unpickle("cifar-10-batches-py/data_batch_1")
b2 = unpickle("cifar-10-batches-py/data_batch_2")
b3 = unpickle("cifar-10-batches-py/data_batch_3")
b4 = unpickle("cifar-10-batches-py/data_batch_4")
b5 = unpickle("cifar-10-batches-py/data_batch_5")

test = unpickle("cifar-10-batches-py/test_batch")

#Preparing test data
test_data = test['data']
test_label = test['labels']
b = np.zeros((10000,10))
b[np.arange(10000),test_label] = 1.0
test_label = b


#Preparing training data
train_data = np.concatenate([b1['data'],b2['data'],b3['data'],b4['data'],b5['data']],axis=0)
train_label = np.concatenate([b1['labels'],b2['labels'],b3['labels'],b4['labels'],b5['labels']],axis=0)
b = np.zeros((50000,10))
b[np.arange(50000),train_label] = 1.0
train_label = b


# Initialize the wieghts of the network
def initialize_weights(dim_layers, STDDEV = 0.001):
    num_of_layers = len(dim_layers) - 1
    weights = {}
    biases = {}
    
    for it in range(1, num_of_layers + 1):
        if (it == 1):
            weights['h1'] = tf.Variable(tf.truncated_normal([dim_layers['input'], dim_layers['h1'] ]))
            biases['b1'] = tf.Variable(tf.zeros([dim_layers['h1']]))
        elif (it == num_of_layers):
            weights['out'] = tf.Variable(tf.truncated_normal([dim_layers['h' + str(it - 1)], dim_layers['out'] ]))
            biases['out'] = tf.Variable(tf.zeros([dim_layers['out']]))
        else :
            weights['h'+ str(it)] = tf.Variable(tf.truncated_normal([dim_layers['h'+ str(it-1)], dim_layers['h'+ str(it)] ]))
            biases['b'+ str(it)] = tf.Variable(tf.zeros([dim_layers['h'+str(it)]]))
    return weights, biases 


# Forward propagation across the network
def DNN(X, weights, biases, dropout_keep_prob):
    num_wgts = len(weights)
    layer = {}
    layer['1'] = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(X, weights['h1']), biases['b1'])), dropout_keep_prob)
    for it in range(2, num_wgts):
        layer[str(it)] = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(layer[str(it - 1)], weights['h'+ str(it)]), biases['b'+ str(it)])), dropout_keep_prob)
        
    return tf.add(tf.matmul(layer[str(num_wgts - 1)], weights['out']), biases['out'])


# Define the constants for the neural network
image_width = 32
image_height = 32
channels = 3
n_classes = 10

# Define the network and optimization paramters for the DNN
dim_layers = {'input': image_width * image_height * channels, 'h1': 1024, 'out': n_classes}
opt_param = dict(lr = 0.001, batch_size = 1024, max_epoch = 100)


# NEM Noise Parameter
noise_mode = 'NEM' # {'Blind', 'NEM', 'None'}
noise_var  = 0.001 # {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3}
ann_factor = 2 # {1, 1.5, 2}

# Optimization Parameter
keep_rate = 1.0
batch_size = opt_param.get('batch_size')
learning_rate = opt_param.get('lr')

# Initialize the weights
weights, biases = initialize_weights(dim_layers, STDDEV = 0.1)
max_iter = int(50000 / opt_param.get('batch_size'))

# Constructing Graph
X = tf.placeholder(tf.float32, [None, image_width * image_height * channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

pred = DNN(X, weights, biases, keep_prob)
prob = tf.nn.softmax(pred)


# Loss, optimizer, and accuracy
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits = pred, labels = y)
    )   # softmax loss
optimizer = tf.train.GradientDescentOptimizer(
            learning_rate = learning_rate
        ).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

pred_tag  = tf.argmax(pred, 1)
label_tag = tf.argmax(y, 1)
###############################################################################################################################

# Function for adding noise to the network

def add_noise(ay, epoch, mode):
    # mode : "NEM", "Blind", or "None"
    # epoch:  Specifies the current epoch
    # ay : is the output activation
    
    nv = noise_var / math.pow(epoch, ann_factor)
    n_classes = dim_layers.get('out')
    batch_size = opt_param.get('batch_size')
    if (mode == "NEM"):
        noise = nv*(np.random.uniform(-0.5,0.5,[batch_size, n_classes])) 
        crit = (noise * np.log(ay + 1e-6)).sum(axis = 1)
        index = (crit >= 0).astype(float)
        noise_index = np.reshape(np.repeat(index, n_classes), [batch_size, n_classes])
        noise_ = noise_index * noise
    elif (mode == "Blind"):
        noise_ = nv*(np.random.uniform(-0.5,0.5,[batch_size, n_classes])) 
    else:
        noise_ = np.zeros((batch_size, n_classes))
    return noise_


acc_test = np.zeros([opt_param.get('max_epoch') + 1])
loss_test = np.zeros([opt_param.get('max_epoch') + 1])


# Train the network
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Train the network
    for epoch in range(opt_param.get('max_epoch') + 1):
        cost_avg = 0.0
        indices = np.random.choice(50000, 50000, replace = False)
        start = 0
        stop  = start + opt_param.get('batch_size')
        for a in range(max_iter):
            batch = indices[start:stop] 
            if (epoch == 0):
                loss1 = sess.run(cost, feed_dict={X: train_data[batch,:], y: train_label[batch,:], keep_prob: 1.0})
            else:
                ay = sess.run(prob, feed_dict ={X: train_data[batch,:], keep_prob : keep_rate})
                # ADD NOISE 
                noise = add_noise(ay, epoch, noise_mode)
                batch_y = train_label[batch,:] + noise
                _, loss1 = sess.run([optimizer,cost], feed_dict={X: train_data[batch,:], y: batch_y, keep_prob: keep_rate})
            
            start = start + opt_param.get('batch_size')
            stop  = stop  + opt_param.get('batch_size')

        test_loss, test_acc = sess.run([cost, accuracy], feed_dict={X: test_data, y: test_label, keep_prob: 1.0})	
        acc_test[epoch] = test_acc
        loss_test[epoch] = tst.loss

        print ("Epoch: %03d/%03d" % (epoch, opt_param.get('max_epoch')))
        print ("Validation accuracy: %.3f" % (test_acc))
        print ("Validation cross-entropy: %.3f" % (test_loss))
    

    ################################## Compute the confusion matrix ##########################################
    confusion_matrix = np.zeros([10, 10])
    pred_ = sess.run(pred_tag, feed_dict = {X: test_data, keep_prob: keep_rate})
    labl_ = sess.run(label_tag, feed_dict = {y: test_label})
    confusion_mat = np.zeros([10, 10])

    for i in range(pred_.shape[0]):
        confusion_mat[labl_[i], pred_[i]] =  confusion_mat[labl_[i], pred_[i]]  +1
    
    sum1 = np.repeat(confusion_mat.sum(axis = 1), [10])
    sum1 = np.reshape(sum1, [10,10])
    confusion_mat = confusion_mat/sum1

    ##################### Report the result
    print("THE CONFUSION MATRIX IS:")
    print(confusion_mat)

    np.savetxt("Test_accuracy.csv", acc_test)
    np.savetxt("Test_loss_fxn.csv", loss_test)
    np.savetxt("Confusion_mat.csv", confusion_mat)
