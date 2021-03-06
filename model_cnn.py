import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
import datetime


# modified Data

def modified_data(file_name):

    raw_input = pd.read_csv(file_name)
    X = raw_input[list(raw_input)[1:]]
    X = X.values.reshape(X.shape[0], 28, 28, 1)
    '''
    contrast_increase = tf.image.adjust_contrast(X, 200)
    with tf.Session() as sess:
        X = sess.run(contrast_increase)
    '''
    Y = pd.get_dummies(raw_input['label'])
    return train_valid(X), train_valid(Y)


def train_valid(input_data, train_portion=0.97):

    num_sample = input_data.shape[0]
    train_num = int(num_sample * train_portion)
    return input_data[:train_num], input_data[train_num:]


# model

def create_placeholders(n_h, n_w, n_c, n_y):

    X = tf.placeholder(tf.float32, [None, n_h, n_w, n_c], name='input')
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y


def initialize_parameters():

    W1 = tf.get_variable('W1', [8, 8, 1, 16], initializer=tf.keras.initializers.he_normal())
    W2 = tf.get_variable('W2', [4, 4, 16, 32], initializer=tf.keras.initializers.he_normal())
    W3 = tf.get_variable('W3', [2, 2, 32, 64], initializer=tf.keras.initializers.he_normal())
    #W4 = tf.get_variable('W4', [2, 2, 64, 128], initializer=tf.keras.initializers.he_normal())
    parameters = {'W1': W1,
                  'W2': W2,
                  'W3': W3,
    #              'W4': W4,
                  }
    return parameters


def forward_propagation(X, parameters, is_training, drop_rate=0.):

    if is_training is False:
        drop_rate = 0.

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    #W4 = parameters['W4']

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.layers.batch_normalization(Z1, training=is_training)
    A1 = tf.nn.relu(A1)

    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.layers.batch_normalization(Z2, training=is_training)
    A2 = tf.nn.relu(A2)

    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding='SAME')
    A3 = tf.layers.batch_normalization(Z3, training=is_training)
    A3 = tf.nn.relu(A3)

    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #Z4 = tf.nn.conv2d(P3, W4, strides=[1,1,1,1], padding='SAME')
    #A4 = tf.layers.batch_normalization(Z4, training=is_training)
    #A4 = tf.nn.relu(A4)

    #P4 = tf.nn.max_pool(A4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    P_out = tf.layers.flatten(P3)

    fc_layer_one = tf.keras.layers.Dense(60, activation=None)
    Z5 = tf.layers.batch_normalization(fc_layer_one(P_out), training=is_training)
    Z5 = tf.nn.relu(Z5)
    Z5 = tf.nn.dropout(Z5, keep_prob=1-drop_rate)

    fc_layer_out = tf.keras.layers.Dense(10, activation=None, use_bias=False)
    Z_out = fc_layer_out(Z5)
    return Z_out


def compute_cost(Z_out, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z_out, labels=Y))
    return cost


def random_mini_batches(X, Y, minibatch_size=64, seed=0):

    n_sample = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(n_sample))
    shuffled_X = X[permutation]
    shuffled_Y = Y.iloc[permutation]

    num_batches = int(n_sample/minibatch_size)

    for n in range(num_batches):
        mini_batch_x = shuffled_X[n * minibatch_size:(n+1) * minibatch_size]
        mini_batch_y = shuffled_Y[n * minibatch_size:(n+1) * minibatch_size]
        mini_batches.append((mini_batch_x, mini_batch_y))

    if n_sample%minibatch_size != 0:
        mini_batch_x = shuffled_X[num_batches * minibatch_size:]
        mini_batch_y = shuffled_Y[num_batches * minibatch_size:]
        mini_batches.append((mini_batch_x, mini_batch_y))

    return mini_batches


def model_cnn(train_x, train_y, valid_x, valid_y, learning_rate=0.0009, num_epoch=100,
              minibatch_size=64, regularization=0, dropout=0., print_cost=True, draw=False):

    (num_sample, n_h, n_w, n_c) = train_x.shape
    n_y = train_y.shape[1]
    costs = []
    valid_costs = []

    X, Y = create_placeholders(n_h, n_w, n_c, n_y)
    is_training = tf.placeholder(tf.bool, name='training')             # for batch normalization
    parameters = initialize_parameters()

# train
    Z_out = forward_propagation(X, parameters, is_training, dropout)
    # compute cost
    cost = compute_cost(Z_out, Y)
    if regularization > 0.:
        regularize = 0
        for key in parameters.keys():
            regularize += tf.nn.l2_loss(parameters[key])
        cost = tf.reduce_mean(cost + regularization * regularize)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # evaluation
    correct_prediction = tf.equal(tf.argmax(Z_out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    predict_result = tf.argmax(Z_out, 1, name='output')
    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)

        if draw is True:
            plt.title('Loss')
            plt.ylabel('loss')
            plt.xlabel('#epoch/5')
            plt.plot(costs, '-rx', label='Train')
            plt.plot(valid_costs, '-bo', label='Test')
            plt.legend(loc=1)
            plt.ion()
            plt.show()

        for epoch in range(num_epoch):

            minibatches = random_mini_batches(train_x, train_y, minibatch_size)
            num_minibatch = len(minibatches)
            epoch_cost = 0
            accuracy_train = 0

            for batch in range(num_minibatch):
                (mini_x, mini_y) = minibatches[batch]
                _, mini_cost = sess.run([optimizer, cost], {X: mini_x, Y: mini_y, is_training: True})
                epoch_cost += mini_cost/num_minibatch
                accuracy_train += accuracy.eval({X: mini_x, Y: mini_y, is_training: False}) / num_minibatch

            if (epoch+1) % 20 == 0 and print_cost is True:
                print('epoch_', epoch+1, ': ', epoch_cost)
            if (epoch+1) % 100 == 0:
                print('[epoch-{0}]\n\tTrain Acc = {1}\tValid Acc = {2}'.format(epoch+1, accuracy_train,
                                                                               accuracy.eval({X: valid_x, Y: valid_y,
                                                                                              is_training: False})))
                saver.save(sess, 'model_cnn_{0}-{1}_r-{2}_d-{3}/model_cnn'.format(learning_rate, num_epoch,
                                                                                  regularization, dropout_rate),
                           global_step=epoch+1)
            if (epoch+1) % 10 == 0:
                costs.append(epoch_cost)
                valid_costs.append(cost.eval({X: valid_x, Y: valid_y, is_training: False}))
            if draw is True and (epoch+1) % 10 == 0:
                plt.plot(costs, '-rx', label='Train')
                plt.plot(valid_costs, '-bo', label='Test')
                plt.pause(0.1)

        # evaluation (cut train to small pieces preventing resource exhausted)
        train_batches_eval = random_mini_batches(train_x, train_y, minibatch_size)
        num_train_batch_eval = len(train_batches_eval)
        accuracy_train = 0
        for train_batch in range(num_train_batch_eval):
            (mini_train_x, mini_train_y) = train_batches_eval[train_batch]
            accuracy_train += accuracy.eval({X: mini_train_x, Y: mini_train_y, is_training: False})/num_train_batch_eval
        print('Final Train Acc: ', accuracy_train)
        print('Final Validation Acc:', accuracy.eval({X: valid_x, Y: valid_y, is_training: False}))
        parameters = sess.run(parameters)
    return parameters


def predict(model_path, model_selected, test):

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        saver = tf.train.import_meta_graph('{0}/model_cnn-{1}.meta'.format(model_path, model_selected))
        saver.restore(sess, '{0}/model_cnn-{1}'.format(model_path, model_selected))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('input:0')
        is_training = graph.get_tensor_by_name('training:0')
        predict_result = graph.get_tensor_by_name('output:0')
        result = []
        part = len(test)//10
        for a in range(9):
            result.extend(sess.run(predict_result, {X: test[a*part:(a+1)*part], is_training: False}))
        result.extend(sess.run(predict_result, {X: test[9*part:], is_training: False}))
        with open('submit_{0}.csv'.format(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")), 'w') as output_file:
            output_file.write('ImageId,Label\n')
            for i in range(len(result)):
                output_file.write('{0},{1}\n'.format(i+1, result[i]))


if __name__ == '__main__':
    opt = argv[1]
    file_input = argv[2]
    if opt == '-tr':
        train_opt = argv[3:]
        epoch_num = 100
        batch_size = 64
        learning_rate = 0.0009
        regularizer = 0
        dropout_rate = 0.
        i = 0
        while i < len(train_opt):
            if train_opt[i] == '-e':
                epoch_num = int(train_opt[i+1])
            elif train_opt[i] == '-b':
                batch_size = int(train_opt[i+1])
            elif train_opt[i] == '-lr':
                learning_rate = float(train_opt[i+1])
            elif train_opt[i] == '-r':
                regularizer = float(train_opt[i+1])
            elif train_opt[i] == '-d':
                dropout_rate = float(train_opt[i+1])
            i += 1

        (train_X, valid_X), (train_Y, valid_Y) = modified_data(file_input)
        train_X = train_X/255
        valid_X = valid_X/255

        learned_parameters = model_cnn(train_X, train_Y, valid_X, valid_Y, learning_rate=learning_rate,
                                       num_epoch=epoch_num, minibatch_size=batch_size, regularization=regularizer,
                                       dropout=dropout_rate, print_cost=True, draw=False)
    elif opt == '-te':
        model_path = argv[2]
        model_selected = argv[3]
        test_file = argv[4]
        real_test_raw = pd.read_csv(test_file)
        num_sample = real_test_raw.shape[0]
        test_X = real_test_raw.values.reshape([num_sample, 28, 28, 1])
        '''
        increase_contrast = tf.image.adjust_contrast(test_X, 200)
        with tf.Session() as sess:
            test_X = sess.run(increase_contrast)
        '''
        test_X = test_X/255
        predict(model_path, model_selected, test_X)
