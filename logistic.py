#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# load both train and test data
train_data = np.genfromtxt('data/project3_train.csv', delimiter=',')
test_data = np.genfromtxt('data/project3_test.csv', delimiter=',')


X_train = train_data[:, 0:3] # features
Y_train = train_data[:, 3] # labels

x_test = test_data[:, 0:3] # features
y_test = test_data[:, 3] # features

np.random.seed(42)
# Add a column of ones to X_train and x_test as the intercept term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

# define the sigmoid function
def sigmoid(val):
    result=1/(1+ np.exp(-val))
    return result

# calculate cost function
def loss_func(x,weights,y):
    no_samples=len(y)
    A=sigmoid(np.dot(x,weights))
    A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
    loss=-(1/no_samples)*np.sum( y*np.log(A_clipped) + (1-y)*np.log(1-A_clipped))
    return loss

# calculate cost functionapplying regularization
def loss_reg_func(x, weights, y, lambda_reg):
    no_samples = len(y)
    A = sigmoid(np.dot(x, weights))
    A_clipped = np.clip(A, 1e-15, 1 - 1e-15) 
    regularization_term = (lambda_reg / (2 * no_samples)) * np.sum(weights[1:]**2) 
    loss = -(1 / no_samples) * np.sum(y * np.log(A_clipped) + (1 - y) * np.log(1 - A_clipped)) + regularization_term
    return loss

#define batch algorithm
def batch_gradient_descent(x,y,iterations,learning_rate):
    no_samples=x.shape[0]
    features=x.shape[1]
    
    weights=np.zeros(features)
    loss_history=[]
    
    for i in range(iterations):
        A=sigmoid(np.dot(x,weights))
        
        dw=(1/no_samples)*np.dot(x.T, (A-y))
        weights=weights-learning_rate*dw
        
        loss=loss_func(x,weights,y)
        loss_history.append(loss)
    
    return weights,loss_history

# stochastic gradient descent
def stochastic_gradient_descent(X, y, weights, learning_rate, iterations, lambda_val):
    start_time = time.time()
    stoch_losses = []

    for m in range(iterations):
        index = np.random.randint(low=0, high=len(y))
        pred = sigmoid(np.dot(X[index], weights))
        dw_s = X[index] * (pred - y[index]) + 2 * lambda_val * weights  # include L2 regularization term
        weights =weights- learning_rate * dw_s

        current_loss = loss_func(X, weights, y) + (lambda_val / 2) * np.sum(weights**2) 
        stoch_losses.append(current_loss)

    finish_time = time.time()
    total_used_time = finish_time - start_time
    print(f"Stochastic_gradient_computational time is: {total_used_time} seconds")

    return weights, stoch_losses

## implementing mini_batch algorithm
def mini_batch_gradient_descent(x, y, learning_rate, batch_size, iterations, lambda_reg):
    start_time=time.time()
    no_samples=x.shape[0]
    features=x.shape[1]
    weights = np.zeros(features)
    losses = []
    
    for iteration in range(iterations):

        for i in range(0, no_samples, batch_size):
            # Calculate the predicted values
            predictions = sigmoid(np.dot(x[i:i+batch_size], weights))

            grad = (np.dot(x[i:i+batch_size].T, (predictions - y[i:i+batch_size])) + lambda_reg * weights) / batch_size
            grad[0] -= lambda_reg * weights[0]  

            grad = np.clip(grad, -1e3, 1e3)
            weights -= learning_rate * grad

        current_loss = loss_reg_func(x, weights, y, lambda_reg)
        losses.append(current_loss)
    finish_time=time.time()
    total_used_time=finish_time-start_time
    print(f'Mini_batched_gradient_computational time is:{total_used_time}')

    return weights, losses

 # define batch algorithm or you can use the one above
def batch_gradient_descent_2(x,y,iterations,learning_rate):
    start_time=time.time()
    
    no_samples=x.shape[0]
    features=x.shape[1]
    batch_weights=np.zeros(features)
    batch_loss_list=[]
    
    for m in range(iterations):
        A=sigmoid(np.dot(x,batch_weights))
        dw=(1/no_samples)*np.dot(x.T, (A-y))
        
        batch_weights=batch_weights-learning_rate*dw
        loss=loss_func(x,batch_weights,y)
        batch_loss_list.append(loss)
        
    finish_time=time.time()
    total_used_time=finish_time-start_time
    print(f'Batched_gradient_computational time is:{total_used_time}')
    
    return batch_weights,batch_loss_list

# function to calculate accuracy
def get_accuracy(x, y, weights):
    predictions = sigmoid(np.dot(x, weights))
    predictions = np.round(predictions)
    accuracy = np.mean(predictions == y)
    return accuracy



learning_rates = [0.001, 0.002, 0.006, 0.01, 0.1]

for lr,color in zip(learning_rates,['black','green','red','cyan','darkblue']):
    weights, loss_history = batch_gradient_descent(X_train, Y_train, 5000, lr )
    plt.plot(range(1, 5001),loss_history,label=f'Learning_rate{lr}',color=color)
    
# Configure plot
plt.title('A GRAPH OF LOSS AGAINST NUMBER OF ITERATIONS')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output/5000_iterations/logistic_regression_loss_plot.jpg')
plt.show()


initial_weights=np.zeros(X_train.shape[1])
lambda_val = 1.0

## call batch stochastic function and print out the accuracy
stoch_final_weights, losses_stoch = stochastic_gradient_descent(X_train, Y_train, initial_weights, 0.1, 300000, lambda_val)
acc_train_stochastic = get_accuracy(X_train, Y_train, stoch_final_weights)
acc_test_stochastic = get_accuracy(x_test, y_test, stoch_final_weights)
print(f"Training data accuracy for stochastic descent: {acc_train_stochastic}")
print(f"Test data accuracy for stochastic descent: {acc_test_stochastic}")

# call mini batch gradient descent
mini_batch_weights, losses_mini = mini_batch_gradient_descent(X_train, Y_train, learning_rate=0.1, batch_size=5, iterations=300000, lambda_reg=0.1)
mini_batch_train_accuracy = get_accuracy(X_train, Y_train, mini_batch_weights)
mini_batch_test_accuracy = get_accuracy(x_test, y_test, weights)
print(f"Training data accuracy for mini batch descent : {mini_batch_train_accuracy}")
print(f"Test data accuracy for mini batch decsent: {mini_batch_test_accuracy}")

## call batch gradient function and print out the accuracy
b_final_weight_batch, loss_batch = batch_gradient_descent_2(X_train, Y_train, learning_rate=0.1, iterations=300000)
acc_train_batch = get_accuracy(X_train, Y_train, b_final_weight_batch)
acc_test_batch = get_accuracy(x_test, y_test, b_final_weight_batch)
print(f"Training data accuracy for batch descent: {acc_train_batch:.4f}")
print(f"Test data accuracy for batch descent: {acc_test_batch:.4f}")




