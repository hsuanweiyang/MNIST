import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sys import argv
import datetime
from tensorflow.contrib.layers import fully_connected


# modified Data

def modified_data(file_name):

    raw_input = pd.read_csv(file_name)
    X = raw_input[list(raw_input)[1:]]
    X = X.values.reshape(X.shape[0], 28, 28, 1)
    Y = pd.get_dummies(raw_input['label'])
    return train_valid(X), train_valid(Y)


def train_valid(input_data, train_portion=0.97):

    num_sample = input_data.shape[0]
    train_num = int(num_sample * train_portion)
    return input_data[:train_num], input_data[train_num:]


# model

def create_placeholders(n_h, n_w, n_c, n_y):

    X = tf.placeholder(tf.float32, [None, n_h, n_w, n_c])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y


def initialize_parameters():

    W1 = tf.get_variable('W1', [8, 8, 1, 8], initializer=tf.keras.initializers.he_normal())
    W2 = tf.get_variable('W2', [4, 4, 8, 16], initializer=tf.keras.initializers.he_normal())
    W3 = tf.get_variable('W3', [2, 2, 16, 32], initializer=tf.keras.initializers.he_normal())
    parameters = {'W1': W1,
                  'W2': W2,
                  'W3': W3,
                  }
    return parameters


def forward_propagation(X, parameters, is_training):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.layers.batch_normalization(Z1, training=is_training)
    A1 = tf.nn.relu(A1)
    Z2 = tf.nn.conv2d(A1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.layers.batch_normalization(Z2, training=is_training)
    A2 = tf.nn.relu(A2)
    P1 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    Z3 = tf.nn.conv2d(P1, W3, strides=[1,1,1,1], padding='SAME')
    A3 = tf.layers.batch_normalization(Z3, training=is_training)
    A3 = tf.nn.relu(A3)
    P2 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    P2 = tf.layers.flatten(P2)
    fc_layer_one = tf.keras.layers.Dense(30, activation=tf.nn.relu, use_bias=True)
    Z3 = fc_layer_one(P2)
    fc_layer_two = tf.keras.layers.Dense(10, activation=None, use_bias=False)
    Z4 = fc_layer_two(Z3)
    return Z4


def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
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


def model_cnn(train_x, train_y, valid_x, valid_y, test_x, learning_rate=0.0009, num_epoch=100,
              minibatch_size=64, regularization=0, print_cost=True, draw=False):

    (num_sample, n_h, n_w, n_c) = train_x.shape
    n_y = train_y.shape[1]
    costs = []
    valid_costs = []

    X, Y = create_placeholders(n_h, n_w, n_c, n_y)
    is_training = tf.placeholder(tf.bool)             # for batch normalization
    parameters = initialize_parameters()

# train
    Z_out = forward_propagation(X, parameters, is_training)
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

    predict_result = tf.argmax(Z_out, 1)

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

            if (epoch+1) % 10 == 0 and print_cost is True:
                print('epoch_', epoch+1, ': ', epoch_cost)
            if (epoch+1) % 100 == 0:
                print('[epoch-{0}]\n\tTrain Acc = {1}\tValid Acc = {2}'.format(epoch+1, accuracy_train,
                                                                               accuracy.eval({X: valid_x, Y: valid_y,
                                                                                              is_training: False})))
            if (epoch+1) % 5 == 0:
                costs.append(epoch_cost)
                valid_costs.append(cost.eval({X: valid_x, Y: valid_y, is_training: False}))
            if draw is True and (epoch+1) % 5 == 0:
                plt.plot(costs, '-rx', label='Train')
                plt.plot(valid_costs, '-bo', label='Test')
                plt.pause(0.1)

        # evaluation (cut train to small pieces preventing resource exhausted)
        train_batches_eval = random_mini_batches(train_x, train_y)
        num_train_batch_eval = len(train_batches_eval)
        accuracy_train = 0
        for train_batch in range(num_train_batch_eval):
            (mini_train_x, mini_train_y) = train_batches_eval[train_batch]
            accuracy_train += accuracy.eval({X: mini_train_x, Y: mini_train_y, is_training: False})/num_train_batch_eval
        print('Final Train Acc: ', accuracy_train)
        print('Final Validation Acc:', accuracy.eval({X: valid_x, Y: valid_y, is_training: False}))

        #output test
        out_result = []
        num_out_test = test_x.shape[0]
        batch_out_test = int(num_out_test/10)
        for i in range(9):
            out_result.extend(predict_result.eval({X: test_x[i * batch_out_test: (i+1) * batch_out_test],
                                                   is_training: False}))
        out_result.extend(predict_result.eval({X: test_x[9 * batch_out_test:], is_training: False}))

        with open('submit_result.csv', 'w') as test_out:
            test_out.write('ImageId,Label\n')
            for i in range(len(out_result)):
                test_out.write('{0},{1}\n'.format(i+1, out_result[i]))
        parameters = sess.run(parameters)
    return parameters

'''
def predict(X, parameters):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Z3 = forward_propagation(x, parameters)
    prediction = tf.argmax(Z3, 1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(prediction, {x: X})
    return result
'''


if __name__ == '__main__':
    opt = argv[1]
    file_input = argv[2]
    if opt == '-tr':
        train_opt = argv[3:]
        epoch_num = 100
        batch_size = 64
        learning_rate = 0.0009
        regularizer = 0
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
            elif train_opt[i] == '-te':
                test_file = train_opt[i+1]
            i += 1

        (train_X, valid_X), (train_Y, valid_Y) = modified_data(file_input)
        train_X = train_X/255
        valid_X = valid_X/255
        real_test_raw = pd.read_csv(test_file)
        num_sample = real_test_raw.shape[0]
        test_X = real_test_raw.values.reshape([num_sample, 28, 28, 1])
        test_X = test_X/255

        learned_parameters = model_cnn(train_X, train_Y, valid_X, valid_Y, test_X, learning_rate=learning_rate,
                                       num_epoch=epoch_num, minibatch_size=batch_size, regularization=regularizer,
                                       print_cost=True, draw=True)
        pickle.dump(learned_parameters, open('cnn_para_ep-{0}_bz-{1}_lr-{2}_r-{3}_{4}'.format(epoch_num, batch_size,
                                                                                        learning_rate, regularizer,
                                                                                        datetime.datetime.now().
                                                                                        strftime('%Y%m%d')), 'wb')
                    )
    '''
    elif opt == '-te':
        raw_input = pd.read_csv(file_input)
        num_sample = raw_input.shape[0]
        X = raw_input.values.reshape(num_sample, 28, 28, 1)
        X = X/255
        parameter_file = argv[3]
        with open(parameter_file, 'rb') as parafile:
            loaded_parameters = pickle.load(parafile)
        predict_result = predict(X, loaded_parameters)
        outfile_name = 'cnn_submit_' + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.csv'
        with open(outfile_name, 'w') as out_file:
            out_file.write('ImageId,Label\n')
            for n in range(len(predict_result)):
                out_file.write('{0},{1}\n'.format(n+1, predict_result[n]))
    '''