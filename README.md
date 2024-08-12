# Neural-Net-Lib
Neural-Net-Lib is a Python library designed to simplify the creation of neural networks. It provides a variety of activation functions, loss functions, and optimization tools that can be easily integrated into your projects.

# Features  
-  Linear Class: A fully connected layer with forward and backward methods.
-  ReLU Activation Class: Implements the ReLU activation function with forward and backward methods.  
-  Tanh Activation Class: Implements the Tanh activation function with forward and backward methods.
-  Sigmoid Activation Class: Implements the Sigmoid activation function with forward and backward methods.  
-  Softmax Activation Class: Implements the Softmax activation function with forward and backward methods.  
-  Cross-Entropy Loss Class: Computes the cross-entropy loss with forward and backward methods.  
-  Mean Squared Error Loss Class: Computes the mean squared error loss with forward and backward methods.  
-  SGD Optimizer Class: Implements stochastic gradient descent for model optimization.  
-  Model Class: A high-level class that wraps layers, loss functions, and optimizers to build and train neural networks.  
-  Mini-Batch Gradient Descent: Supports training with mini-batches for improved efficiency.  
  
# Example usage    
from neural_net_lib import Model, CrossEntropyLoss, SGD, Linear, ReLU, Softmax  
model = Model()  
model.add_layer(Linear(784, 128))  
model.add_layer(ReLU())  
model.add_layer(Linear(128, 10))  
model.add_layer(Softmax())  
loss = CrossEntropyLoss()  
optimizer = SGD(learning_rate=0.01)  
model.compile(loss, optimizer)  
model.train(x_train, y_train, epochs=20, batch_size=64)  
test_loss, test_accuracy = model.evaluate(x_test, y_test)  
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')  

# Installation  
Run this in your terminal to install  
git clone https://github.com/Abbas2905/Neural-Net.git  
(This will create a folder with the code for the library in it)  

# Contact  
For any questions or issues, please open an issue on GitHub or contact  at abbascodes23@gmail.com.
