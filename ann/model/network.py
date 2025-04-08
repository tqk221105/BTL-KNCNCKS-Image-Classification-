import numpy as np


# network: List of layers in the network
# loss: Function is used to calculate the loss
# loss_prime: Derivative of loss
# x_train: Input data
# y_train: Output Label
# epoch: Number of training.
# learning_rate
# verbose: If true show


def train(network, loss, loss_prime, x_train, y_train, epoch=1000, learning_rate=0.01, verbose=True):
    for e in range(epoch):
        error = 0
        for x, y in zip(x_train, y_train):
            output = x

            # Forward propagation
            for layer in network:
                output = layer.forward(output)

            # Compute loss
            error += loss(y, output)

            # Backward propagation
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # Show efficiency
        if verbose and (e + 1) % 100 == 0:
            print(f"Epoch {e+1}/{epoch}, Error = {error / len(x_train):.6f}")



def predict(network, input_data):
    results = []
    for x in input_data:
        output = x
        for layer in network:
            output = layer.forward(output)
        results.append(output)
    return np.array(results)
    
