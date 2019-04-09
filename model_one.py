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
    X = pd.DataFrame.transpose(raw_input[list(raw_input)[1:]])
    Y = pd.DataFrame.transpose(pd.get_dummies(raw_input['label']))
    return train_valid_test(X), train_valid_test(Y)


def train_valid_test(input_data, train_portion=0.95):

    num_sample = input_data.shape[1]
    train_num = int(num_sample * train_portion)
    valid_num = int((num_sample - train_num)/2)
    return input_data.iloc[:, :train_num], input_data.iloc[:, train_num:train_num + valid_num], \
           input_data.iloc[:, train_num + valid_num:]

# model


def create_placeholder(n_features, n_y):

    X = tf.placeholder(tf.float32, [n_features, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    return X, Y


def initialize_parameters(n_feature):

    W1 = tf.get_variable('W1', [30, n_feature], initializer=tf.keras.initializers.he_normal())
    b1 = tf.get_variable('b1', [30, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [20, 30], initializer=tf.keras.initializers.he_normal())
    b2 = tf.get_variable('b2', [20, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [10, 20], initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))
    b3 = tf.get_variable('b3', [10, 1], initializer=tf.zeros_initializer())
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    return parameters


def add_layer(W, X, b, act_fun=tf.nn.relu):

    Z = tf.matmul(W, X) + b
    A = act_fun(Z)
    return A


def forward_propagation(X, parameters, drop_rate):

    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    A1 = add_layer(W1, X, b1)
    A1 = tf.nn.dropout(A1, rate=drop_rate)
    A2 = add_layer(W2, A1, b2)
    A2 = tf.nn.dropout(A2, rate=drop_rate)

    Z3 = tf.matmul(W3, A2) + b3
    return Z3


def compute_cost(Y, Z3):

    Y = tf.transpose(Y)
    Z3 = tf.transpose(Z3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z3))
    return loss


def random_mini_batches(X, Y, minibatch_size=64, seed=0):

    n_sample = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(n_sample))
    shuffled_X = X.iloc[:, permutation]
    shuffled_Y = Y.iloc[:, permutation]

    num_batches = int(n_sample/minibatch_size)

    for n in range(num_batches):
        mini_batch_x = shuffled_X.iloc[:, n * minibatch_size:(n+1) * minibatch_size]
        mini_batch_y = shuffled_Y.iloc[:, n * minibatch_size:(n+1) * minibatch_size]
        mini_batches.append((mini_batch_x, mini_batch_y))

    if n_sample%minibatch_size != 0:
        mini_batch_x = shuffled_X.iloc[:, num_batches * minibatch_size:]
        mini_batch_y = shuffled_Y.iloc[:, num_batches * minibatch_size:]
        mini_batches.append((mini_batch_x, mini_batch_y))

    return mini_batches


def model_fc(train_X, train_Y, valid_X, valid_Y, dropout, learning_rate=0.0001, num_epoch=1500,
             minibatch_size=64, regularize=False, print_cost=True, draw=False):

    (n_features, m) = train_X.shape
    n_y = train_Y.shape[0]
    costs = []
    valid_costs = []

# train
    X, Y = create_placeholder(n_features, n_y)
    parameters = initialize_parameters(n_features)
    Z3 = forward_propagation(X, parameters, dropout)
    cost = compute_cost(Y, Z3)
    reg = '0'   # regularization status
    if regularize:
        reg = '1'
        regularizer = 0
        for key in parameters.keys():
            regularizer += tf.nn.l2_loss(parameters[key])
        lamda = 0.02
        cost = tf.reduce_mean(cost + lamda * regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # eval
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# valid
    Z3_test = forward_propagation(X, parameters, 0)
    cost_test = compute_cost(Y, Z3_test)
    # eval
    correct_prediction_test = tf.equal(tf.argmax(Z3_test), tf.argmax(Y))
    accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

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
            epoch_cost = 0.
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size)
            num_minibatches = len(minibatches)
            
            for minibatch in minibatches:
                (mini_x, mini_y) = minibatch
                _, mini_cost = sess.run([optimizer, cost], {X: mini_x, Y: mini_y})

                epoch_cost += mini_cost/num_minibatches

            if (epoch+1) % 100 == 0 and print_cost is True:
                print('epoch_', epoch+1, ': ', epoch_cost)
            if (epoch+1) % 5 == 0:
                costs.append(epoch_cost)
                valid_costs.append(cost_test.eval({X: valid_X, Y: valid_Y}))
            if draw is True and (epoch+1) % 5 == 0:
                plt.plot(costs, '-rx', label='Train')
                plt.plot(valid_costs, '-bo', label='Test')
                plt.pause(0.1)
        plt.savefig('fig_ep-{0}_bz-{1}_lr-{2}_reg-{3}_dr-{4}_{5}.png'.format(num_epoch,
                                                                  minibatch_size,
                                                                  learning_rate, reg, dropout,
                                                                  datetime.datetime.now().strftime('%Y%m%d')
                                                                  )
                    )
        print('Final Train Acc:', accuracy.eval({X: train_X, Y: train_Y}))
        print('Final Validation Acc:', accuracy_test.eval({X: valid_X, Y: valid_Y}))
        parameters = sess.run(parameters)

    return parameters


def predict(X, parameters):

    x = tf.placeholder(tf.float32, [X.shape[0], None])
    Z3 = forward_propagation(x, parameters, 0)
    prediction = tf.argmax(Z3)

    with tf.Session() as sess:
        result = sess.run(prediction, {x:X})

    return result


if __name__ == '__main__':
    file_input = argv[1]
    opt = argv[2]

    if opt == '-tr':
        train_opt = argv[3:]
        epoch_num = 1500
        batch_size = 64
        learning_rate = 0.0001
        regularization = False
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
                regularization = True
            elif train_opt[i] == '-d':
                dropout_rate = float(train_opt[i+1])
            i += 1

        (train_X, valid_X, test_X), (train_Y, valid_Y, test_Y) = modified_data(file_input)
        train_X = train_X/255
        valid_X = valid_X/255
        test_X = test_X/255
        learned_para = model_fc(train_X, train_Y, valid_X, valid_Y, dropout_rate,
                                num_epoch=epoch_num, minibatch_size=batch_size,
                                regularize=regularization, learning_rate=learning_rate,
                                draw=True)
        pickle.dump(learned_para, open('fc_para_ep-{0}_bz-{1}_lr-{2}_reg-{3}_dr-{4}_{5}.pkl'.
                                       format(epoch_num, batch_size, learning_rate,
                                       int(regularization), dropout_rate,
                                        datetime.datetime.now().strftime('%Y%m%d')), 'wb'))
        predict_result = predict(test_X, learned_para)

    elif opt == '-te':
        raw_data = pd.read_csv(file_input)
        X = pd.DataFrame.transpose(raw_data)
        parameter_file = argv[3]
        with open(parameter_file, 'rb') as file:
            loaded_parameter = pickle.load(file)
        predict_result = predict(X, loaded_parameter)
        outfile_name = 'submit_' + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.csv'
        with open(outfile_name, 'w') as outfile:
            outfile.write('ImageId,Label\n')
            for n in range(len(predict_result)):
                outfile.write('{0},{1}\n'.format(n+1, predict_result[n]))

