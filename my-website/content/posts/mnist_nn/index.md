+++
title = 'From Zero to Predicting Digits: Coding a Neural Network by Hand'
date = 2024-09-18T15:51:45-06:00
draft = false
+++

Welcome to my very first post of the blog! I wanted to take some time to brush up on ML foundations now that I'm in between jobs, and what better place to start than popular computer vision tasks? Knowing myself, I have a habit of not always completing projects that I begin, so my hope is that treating blog posts as completion artifacts for these projects will be a useful forcing function for seeing things through.

Into the meaty content. In this post, I will walk through the implementation of a simple fully-connected neural network to tackle image classification on the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), which contains 70,000 28x28 pixel images of handwritten digits. I will implement backpropagation and stochastic gradient descent from scratch using `numpy` and provide high-level derivations and intuition for computing weight updates of each of the neurons, but I'll try not to get overly academic with it. This was a fun and surprisingly challenge exercise, and it made me even more thankful that mature automatic differentiation libraries like `pytorch` exist - I imagine that manually computing gradients for a 30+ layer ResNet would entail a special kind of masochism.

<!-- ## Loading the Dataset

Let's start off by loading the train and test datasets from the `torchvision` python package. While we won't be using `pytorch` for training, we will make use of the `DataLoader` class for sampling minibatches from the training dataset.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # Ensure fast so no action is needed
])

# Fetch the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```
 -->

## MNIST Digit Classification

Let's begin by giving a brief overview of the MNIST dataset and image classification task. As mentioned, MNIST is comprised of 28x28 pixel images of handwritten digits, and is divided into 60,000 training images and 10,000 test images.

Each image has a corresponding label which is a real number in the range $[0, 9]$. Naturally, the task will be to design an algorithm which is able to correctly classify as many images in our dataset (or more precisely, our test set) correctly.

#### Sample images with corresponding labels:

![MNIST sample](images/mnist_sample_with_labels.png)

## Neural Network Overview

Let's first begin by sa

Our neural network will consist of a single hidden layer, where each node in the hidden layer applies an activation function to a weighted sum of the inputs. The choice of activation function is crucial, as it introduces non-linearity to the model, enabling it to learn complex patterns.

TODO: define mathematically.

In this example, we’ll implement a fully connected (FC) network using Python and NumPy. We initialize random weights for each layer and choose the sigmoid function as the activation function. Feel free to swap it out with others like ReLU or tanh, depending on the task.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class FCNetwork():
    """Single hidden layer network"""
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01 # d x h
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.01 # h x 10
        self.b1 = np.zeros((1, hidden_dim)) # 1 x h
        self.b2 = np.zeros((1, output_dim)) # 1 x 10
        self.activation = activation
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

Here, our network's forward propagation step computes the output probabilities using the softmax function. We’ll use a sigmoid function in the hidden layer for non-linearity, and softmax at the output to compute probabilities for each class (since this is a classification problem).

## Defining the Loss Function

For our classification task, we’ll use the **cross-entropy loss**, which is a common choice for multi-class classification problems. It measures the difference between the predicted probability distribution and the true distribution (one-hot encoded labels for MNIST).

### Python Code for Cross-Entropy Loss:
```python
def cross_entropy_loss(y, y_hat):
    # Small epsilon added to avoid log(0)
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)  # Ensure y_hat is within (0, 1) to prevent log(0)
    
    # Average over the batch
    return -np.sum(y * np.log(y_hat)) / y.shape[0]
```

In this function, we first clip the predicted values y_hat to avoid undefined values from log(0) and then compute the average loss over the batch of examples.

## Implementing Backpropagation

Next up, we implement **backpropagation**, which is the algorithm that allows the model to update its weights based on the gradient of the loss function with respect to each parameter. This is done using the chain rule of calculus to propagate the error from the output layer back to the input layer.

TODO: derive update rules

### Python Code for Backpropagation:
```python
def backprop(X, y, model, learning_rate=0.01):
    # Forward pass
    y_hat = model.forward(X)
    
    # Compute the error at the output layer
    dz2 = y_hat - y  # (batch_size, 10)
    dw2 = np.dot(model.a1.T, dz2) / X.shape[0]
    db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]
    
    # Compute the error at the hidden layer
    dz1 = np.dot(dz2, model.w2.T) * sigmoid_derivative(model.z1)
    dw1 = np.dot(X.T, dz1) / X.shape[0]
    db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]
    
    # Update weights and biases
    model.w2 -= learning_rate * dw2
    model.b2 -= learning_rate * db2
    model.w1 -= learning_rate * dw1
    model.b1 -= learning_rate * db1
```

The backpropagation algorithm updates the weights (w1 and w2) and biases (b1 and b2) by computing the gradients of the loss with respect to each parameter. These gradients are used to adjust the parameters in the direction that reduces the loss, as governed by the learning rate.

## Evaluating Performance

After training the model, we want to evaluate how well it generalizes to unseen data (our test set). The accuracy metric is a simple yet effective measure, especially for classification tasks like MNIST.

### Python Code for Accuracy:
```python
def accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_true_labels == y_pred_labels)
```

Here, we convert the one-hot encoded labels and predictions into their respective class indices using argmax, and then compute the percentage of correctly predicted examples.

## Training the Model

We can now tie everything together in a training loop. The model will iterate over the training data, compute the loss, backpropagate the errors, and update its parameters.

### Python Code for Training Loop:

```python
epochs = 10
learning_rate = 0.01

for epoch in range(epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.view(X_batch.size(0), -1).numpy()  # Flatten the input images
        y_batch_onehot = np.eye(10)[y_batch.numpy()]  # Convert labels to one-hot encoding
        
        # Forward and Backpropagation
        backprop(X_batch, y_batch_onehot, model, learning_rate)
    
    # Test performance on test set
    test_X = test_loader.dataset.data.view(-1, 28*28).numpy()
    test_y = np.eye(10)[test_loader.dataset.targets.numpy()]
    test_predictions = model.forward(test_X)
    test_accuracy = accuracy(test_y, test_predictions)
    print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.4f}")
```

This loop trains the model for a set number of epochs, where each epoch processes the entire dataset. After each epoch, we compute the accuracy on the test dataset.

## Debugging

Training a neural network from scratch can often result in a few hiccups along the way, including issues like vanishing gradients, slow convergence, or poor generalization. A few debugging tips:

Check the learning rate: If the model is not improving, the learning rate may be too high or too low.
Inspect gradients: If the weights are not updating properly, inspect the gradients and make sure they are neither too large nor vanishingly small.
Try different activations: Sigmoid can suffer from saturation in deep networks. Experiment with ReLU or Leaky ReLU if needed.

## Conclusion

In this post, we’ve implemented a fully connected neural network from scratch using NumPy, trained it using stochastic gradient descent and backpropagation, and tested it on the MNIST dataset. This foundational understanding will be useful as we move to more advanced architectures.

Next, we’ll take on the challenge of implementing a **convolutional neural network (CNN)** to tackle a more complex dataset, the CIFAR-10, where image recognition becomes more nuanced.

Stay tuned!