import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sys import argv
import datetime


# modified Data

def modified_data(file_name):

    raw_input = pd.read_csv(file_name)
    X = raw_input[list(raw_input)[1:]]
    X = X.values.reshape(X.shape[0], 28, 28, 1)
    Y = pd.get_dummies(raw_input['label'])
    return train_valid_test(X), train_valid_test(Y)


def train_valid_test(input_data, train_portion=0.95):

    num_sample = input_data.shape[0]
    train_num = int(num_sample * train_portion)
    valid_num = int((num_sample - train_num)/2)
    return input_data[:train_num], input_data[train_num:train_num + valid_num], \
           input_data[train_num + valid_num:]


# model

def create_placeholders(n_h, n_w, n_c, n_y):

    X = tf.placeholder(tf.float32, [None, n_h, n_w, n_c])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y


def initialize_parameters():

    W1 = tf.get_variable('W1', [4, 4, 1, 8], initializer=tf.keras.initializers.he_normal())
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.keras.initializers.he_normal())
    parameters = {'W1': W1,
                  'W2': W2
                  }
    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    P2 = tf.layers.flatten(P2)
    fc_layer = tf.keras.layers.Dense(10, activation=None, use_bias=True)
    Z3 = fc_layer(P2)
    return Z3


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


def model_cnn(train_x, train_y, valid_x, valid_y, learning_rate=0.0009, num_epoch=100,
              minibatch_size=64, regularization=0, print_cost=True, draw=False):

    (num_sample, n_h, n_w, n_c) = train_x.shape
    n_y = train_y.shape[1]
    costs = []
    valid_costs = []

    X, Y= create_placeholders(n_h, n_w, n_c, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    if regularization > 0.:
        regularize = 0
        for key in parameters.keys():
            regularize += tf.nn.l2_loss(parameters[key])
        cost = tf.reduce_mean(cost + regularization * regularize)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        if draw is True:
            plt.title('Loss')
            plt.ylabel('loss')
            plt.xlabel('#epoch/2')
            plt.plot(costs, '-rx', label='Train')
            plt.plot(valid_costs, '-bo', label='Test')
            plt.legend(loc=1)
            plt.ion()
            plt.show()

        for epoch in range(num_epoch):

            minibatches = random_mini_batches(train_x, train_y, minibatch_size)
            num_minibatch = len(minibatches)
            epoch_cost = 0

            for batch in range(num_minibatch):
                (mini_x, mini_y) = minibatches[batch]
                _, mini_cost = sess.run([optimizer, cost], {X:mini_x, Y:mini_y})
                epoch_cost += mini_cost/num_minibatch

            if (epoch+1) % 5 == 0 and print_cost is True:
                print('epoch_', epoch+1, ': ', epoch_cost)
            if (epoch+1) % 2 == 0:
                costs.append(epoch_cost)
                valid_costs.append(cost.eval({X:valid_x, Y: valid_y}))
            if draw is True and (epoch+1) % 2 == 0:
                plt.plot(costs, '-rx', label='Train')
                plt.plot(valid_costs, '-bo', label='Test')
                plt.pause(0.1)

        # evaluation
        train_batches_eval = random_mini_batches(train_x, train_y)
        num_train_batch_eval = len(train_batches_eval)
        accuracy_train = 0
        for train_batch in range(num_train_batch_eval):
            (mini_train_x, mini_train_y) = train_batches_eval[train_batch]
            accuracy_train += accuracy.eval({X:mini_train_x, Y:mini_train_y})/num_train_batch_eval
        print('Final Train Acc: ', accuracy_train)
        print('Final Validation Acc:', accuracy.eval({X: valid_x, Y: valid_y}))
        parameters = sess.run(parameters)
    return parameters


def predict(X, parameters):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Z3 = forward_propagation(x, parameters)
    prediction = tf.argmax(Z3, 1)
    with tf.Session() as sess:
        result = sess.run(prediction, {x: X})
    return result


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
            i += 1

        (train_X, valid_X, test_X), (train_Y, valid_Y, test_Y) = modified_data(file_input)
        train_X = train_X/255
        valid_X = valid_X/255
        test_X = test_X/255

        learned_parameters = model_cnn(train_X, train_Y, valid_X, valid_Y, learning_rate=learning_rate,
                                       num_epoch=epoch_num, minibatch_size=batch_size, regularization=regularizer,
                                       print_cost=True, draw=True)
        pickle.dump(learned_parameters, open('cnn_para_ep-{0}_bz-{1}_lr-{2}_r-{3}_{4}'.format(epoch_num, batch_size,
                                                                                        learning_rate, regularizer,
                                                                                        datetime.datetime.now().
                                                                                        strftime('%Y%m%d')), 'wb')
                    )
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
                out_file.write('{0},{1}'.format(n+1, predict_result[n]))