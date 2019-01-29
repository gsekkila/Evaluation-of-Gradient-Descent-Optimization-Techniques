import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from load_mnist import mnist
import matplotlib.pyplot as pl

def initialize_multilayer_weights(net_dims):
    
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.normal(0, np.sqrt(2.0/net_dims[l]), (net_dims[l+1],net_dims[l]))
        parameters["b"+str(l+1)] = np.random.normal(0, np.sqrt(2.0/net_dims[l]), (net_dims[l+1],1))
    return parameters

def relu(Z):
 
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):

    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    
    intY=Y.astype(int)
    Y_list = intY.tolist()

    A= np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)),axis=0, keepdims=True)

    cache = {}
    cache["A"] = A
    if Y.shape[0] == 0:
        loss = []
    else:
        one_hot_Y = one_hot(Y_list,10)
        Y_fin = one_hot_Y.T
        loss = -np.sum(Y_fin*np.log(A+1e-9))/A.shape[1]

    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    intY=Y.astype(int)
    Y_list = intY.tolist()

    one_hot_Y = one_hot(Y_list,10)
    Y_fin = one_hot_Y.T
    A = cache["A"]
    
    dZ = A - Y_fin
    return dZ

def linear_forward(A, W, b):
    
    Z = np.dot(W,A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
        
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    A_prev = cache["A"]
    dA_prev = np.dot(W.T,dZ)
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ,keepdims=True,axis=1)/m
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
   
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
   
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    Y_onehot = np.zeros((len(x), n))

    Y_onehot[np.arange(len(x)), (x)] = 1
    return Y_onehot

def classify(X, parameters):
    A_L, caches = multi_layer_forward(X, parameters)
    A_smax, smax_cache, cost = softmax_cross_entropy_loss(A_L)

    Ypred = np.argmax(A_smax,axis = 0)
    Ypred = np.reshape(Ypred,(1,Ypred.shape[0]))

    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    for i in range(L):
        parameters["W" + str(i+1)] -= alpha * gradients["dW" + str(i+1)]
        parameters["b" + str(i+1)] -= alpha * gradients["db" + str(i+1)]

    return parameters, alpha

def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01, batch_size = 1000):

    X_val, Y_val, X_train, Y_train = None, None, None, None
    for i in range(10):
        X_temp, Y_temp = X[:, i*600:(i+1)*600], Y[:, i*600:(i+1)*600]
        
        if X_train is None:
            X_train, Y_train = X_temp[:, :500], Y_temp[:, :500]
            X_val, Y_val = X_temp[:, 500:600], Y_temp[:, 500:600]    
        else:
            X_train, Y_train = np.concatenate((X_train, X_temp[:, :500]), axis=1), np.concatenate((Y_train, Y_temp[:, :500]), axis=1)
            X_val, Y_val = np.concatenate((X_val, X_temp[:, 500:600]), axis=1), np.concatenate((Y_val, Y_temp[:, 500:600]), axis=1)

    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    num_classes = 10
    alpha = learning_rate
    val_loss = []
    batches_x = np.split(X,batch_size,axis=1)
    batches_y = np.split(Y,batch_size,axis=1)
    #Y_one_hot = one_hot(Y,num_classes)
    n_batches = len(batches_x)
    for ii in range(num_iterations):
        for index in range(n_batches):
            # Forward Prop
            ## call to multi_layer_forward to get activations
            Z, caches = multi_layer_forward(batches_x[index], parameters)

            ## call to softmax cross entropy loss
            A_L, smax_cache, cost = softmax_cross_entropy_loss(Z, batches_y[index])

            # Backward Prop
            ## call to softmax cross entropy loss der
            dZ = softmax_cross_entropy_loss_der(batches_y[index], smax_cache)
            ## call to multi_layer_backward to get gradients
            gradients = multi_layer_backward(dZ, caches, parameters)

            ## call to update the parameters
            parameters,alpha = update_parameters(parameters, gradients, ii, learning_rate,decay_rate)
        costs.append(cost)
        # if ii % 10 == 0:
        A_L, caches = multi_layer_forward(X_val, parameters)
        A_L, smax_cache, validation_loss = softmax_cross_entropy_loss(A_L, Y_val)

        val_loss.append(validation_loss)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
            print("Validation loss at iteration %i: %f" %(ii, validation_loss))
            
    train_Pred = classify(X_train,parameters)
    
    trAcc = train_Pred - Y_train
    
    trAcc[trAcc != 0] = 1
   
    
    trAcc = ( 1 - np.count_nonzero(train_Pred - Y_train ) / float(train_Pred.shape[1])) * 100 
    # teAcc = ( 1 - np.count_nonzero(test_Pred - test_label ) / float(test_Pred.shape[1]) ) * 100

   
            
    return costs, val_loss, parameters
        
def main():
    n_in, m = 784,784
    n_h1 = 500
    n_h2 = 100
    net_dims = [n_in, n_h1, n_h2]
    #net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
            mnist(noTrSamples=6000,noTsSamples=1000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=600, noTsPerClass=100)

    learning_rate_array = [0.01, 0.07, 0.1]
    num_iterations = 300
    batch_size = len(train_label)
    plots = []


    for learning_rate in learning_rate_array:

        costs, val_loss, parameters = multi_layer_network(train_data, train_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)

        # compute the accuracy for training set and testing set
        train_Pred = classify(train_data, parameters)
        test_Pred = classify(test_data, parameters)

        test_error = ((test_Pred[0]!=test_label[0]).sum())/ test_Pred[0].sum()
        print("Test error: {0:0.4f} ".format(test_error))

        trAcc = train_Pred - train_label
        teAcc = test_Pred - test_label
        trAcc[trAcc != 0] = 1
        teAcc[teAcc != 0] = 1

        trAcc = ( 1 - np.count_nonzero(train_Pred - train_label ) / float(train_Pred.shape[1])) * 100
        teAcc = ( 1 - np.count_nonzero(test_Pred - test_label ) / float(test_Pred.shape[1]) ) * 100

        print("Accuracy for training set is {0:0.3f} %".format(trAcc))
        print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

        # points = np.arange(0, 300)
        # plot_train, = plt.plot(costs[:len(costs)])
        # # plots.append(plot_train)
        plot_val, = plt.plot(val_loss)
        plots.append(plot_val)
        plt.xlabel("No Of Iterations")
        plt.ylabel("Cost/ val_loss")
        plt.title("Loss vs Number of Iterations with varying Learning rates")

    plt.legend(plots,learning_rate_array)
    plt.savefig("Loss plot for No Momentum")

if __name__ == "__main__":
    main()