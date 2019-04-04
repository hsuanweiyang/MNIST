import tensorflow as tf
import numpy as np
import pandas as pd

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


def forward_propagation(X, parameters):

    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    A1 = add_layer(W1, X, b1)
    A2 = add_layer(W2, A1, b2)
    Z3 = tf.matmul(W3, A2)
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


def model(train_X, train_Y, valid_X, valid_Y, learning_rate=0.0001, num_epoch=1500,
          minibatch_size=64, print_cost=True):

    (n_features, m) = train_X.shape
    n_y = train_Y.shape[0]
    costs = []

    X, Y = create_placeholder(n_features, n_y)
    parameters = initialize_parameters(n_features)

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Y, Z3)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # eval
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epoch):
            epoch_cost = 0.
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size)
            num_minibatches = len(minibatches)
            
            for minibatch in minibatches:
                (mini_x, mini_y) = minibatch
                _, mini_cost = sess.run([optimizer, cost], {X: mini_x, Y: mini_y})

                epoch_cost += mini_cost/num_minibatches

            if epoch % 100 == 0 and print_cost is True:
                print('epoch_', epoch, ': ', epoch_cost)
                print('Train Acc:', accuracy.eval({X: train_X, Y: train_Y}))
                print('Validation Acc:', accuracy.eval({X: valid_X, Y: valid_Y}))
            if epoch % 5 == 0 and print_cost is True:
                costs.append(epoch_cost)
        print('Final Train Acc:', accuracy.eval({X: train_X, Y: train_Y}))
        print('Final Validation Acc:', accuracy.eval({X: valid_X, Y: valid_Y}))
        parameters = sess.run(parameters)

    return parameters


(train_X, valid_X, test_X), (train_Y, valid_Y, test_Y) = modified_data('train.csv')

train_X = train_X/255
valid_X = valid_X/255
test_X = test_X/255

model(train_X, train_Y, valid_X, valid_Y, num_epoch=1000)
