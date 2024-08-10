import numpy as np

class Linear:
    """
    A linear (fully connected) layer of a neural network.

    Attributes:
    
    weights : np.ndarray
        Weight matrix of shape (n_neurons, n_inputs).
    biases : np.ndarray
        Bias vector of shape (n_neurons, 1).
    input : np.ndarray
        Input data of shape (n_inputs, m).
    output : np.ndarray
        Output data of shape (n_neurons, m).
    m : int
        Number of examples (columns) in the input data.
    """

    def __init__(self,n_inputs,n_neurons):
        """
        Initializes linear layer with random weights and zero biases
        
        Args:
        n_inputs : int
            number of input features
         n_neurons : int
            number of neurons in the layer
        """


        self.weights=np.random.randn(n_neurons,n_inputs)*0.10
        self.biases=np.zeros((n_neurons,1))
        

    def forward(self,X):
        """
        Performs forward pass of the layer

        Args:

        X : np.ndarray
            Input of the layer of shape (n_inputs,m)

        Returns:
        np.ndarray
           Output of the layer of shape (n_neurons,m)

        
        """
        self.input=X
        self.m=X.shape[1]
        self.output=np.dot(self.weights,X) + self.biases
        return self.output 
    
    def backward(self,dz):
        """
        Performs backward pass of the layer
       
        Args:

        dz : np.ndarray
            Gradient of loss wrt output of the layer

        Returns: 
        np.ndarray
            Gradient of loss wrt input of the layer
        """
        self.dweights=np.dot(dz,self.input.T)/self.m
        self.dbiases=np.sum(dz,axis=1,keepdims=True)/self.m
        return np.dot(self.weights.T,dz)
        

class ReLU:
    """
    Rectified Linear Unit(ReLU) activation function
    """
    def forward(self,X):
        """
        Performs the forward pass

        Args:

        X : np.ndarray 
           Input data 

        Returns :
        np.ndarray
            Output data after applying ReLU          
        
        """
        self.input=X
        return np.maximum(0,X)
 
    def backward(self,gradient):
        """
        Performs the backward pass

        Args:
        gradient : np.ndarray
                Gradient of loss wrt output
       
        Returns :
        np.ndarray
                Gradient of loss wrt input
              
        """
        return gradient*(self.input>0)


class Sigmoid:
    """
    Sigmoid Activation Function
    """
    def forward(self,X):
        """
        Performs the forward pass

        Args:

        X : np.ndarray 
           Input data 

        Returns :
        np.ndarray
            Output data after applying Sigmnoid   
        
        """
        self.input=X
        self.output=1/(1+np.exp(-X))
        return self.output
    
    def backward(self,gradient):
        
        """
        Performs the backward pass

        Args:
        gradient : np.ndarray
                Gradient of loss wrt output
       
        Returns :
        np.ndarray
                Gradient of loss wrt input
             
        """
        return self.output*(1-self.output)*gradient
    

class Tanh:
    """
    Hyperbolic tangent (tanh) activation function.
    """
    def forward(self,X):
        """
        Performs the forward pass

        Args:

        X : np.ndarray 
           Input data 

        Returns :
        np.ndarray
            Output data after applying Tanh   
        
        """
        self.input=X
        self.output=np.tanh(X)
        return self.output
    
    def backward(self,gradient):
        """
        Performs the backward pass

        Args:
        gradient : np.ndarray
                Gradient of loss wrt output
       
        Returns :
        np.ndarray
                Gradient of loss wrt input
             
        """
        return gradient*(1-self.output**2)
    
class Softmax:
    """
    Softmax Activation Function
    """
    def forward(self,X):
        """
        Performs the forward pass

        Args:
        X : np.ndarray
            Input data

        Returns:
        np.ndarray
            Output data after applying softmax

        """
        self.input=X
        X_norm = X - np.max(X, axis=0, keepdims=True)
        exp_values = np.exp(X_norm)
        self.output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return self.output 

    def backward(self,gradient):
        """
        Performs backward pass

        Args:
        gradient : np.ndaray
            Gradient of loss wrt output

        Returns:
        np.ndarray
            Gradient of loss wrt input
        """
        return gradient

class CrossEntropyLoss:
    """
    Cross-Entropy loss function
    """


    def one_hot(self,y_true):
            """
            Converts integer labels to one-hot encoded labels

            Args:
            y_true : np.ndarray
                   Array of true labels
            
            Returns:
            np.ndarray
                   One-hot encoded label matrix
            """
            num_cat=np.max(y_true)+1
            b=np.zeros((num_cat,y_true.shape[1]))
            b[y_true,np.arange(y_true.shape[1])]=1
            return b

        
        
    def forward(self,y_pred,y_true):
        """
        Computes forward pass of the loss
        
        Args:
        y_pred : np.ndarray
               Predicted probabilities
        
        y_true : np.ndarray
                True labels

        Returns: 
        float 
           Computed loss value
        
        """
        m=y_true.shape[1]
        self.y_pred=np.clip(y_pred,1e-7,1-(1e-7))
     
        self.y_true=self.one_hot(y_true)
      
        self.loss=-np.sum(self.y_true*np.log(self.y_pred))
        self.loss=self.loss/m
        return self.loss
    
    def backward(self):
        """
        Computes backward pass of the loss

        Returns:
        np.ndarray 
             Gradient of loss wrt predicted output

        """
     
        return (self.y_pred-self.y_true)/self.y_pred.shape[1]


class MSE:
    """
    Mean Sqaured Error (MSE) loss function
    """

    def forward(self,y_pred,y_true):
        """
        Computes forward pass of the loss

        Args:
        y_pred : np.ndarray
               Predicted values
        
        y_true : np.ndarray
                True values
        
        Returns: 
        float 
           Computed loss value

        """
        self.y_pred=y_pred
        self.y_true=y_true
        self.diff=y_pred-y_true

        return np.sum(np.square(self.diff))/y_pred.shape[1]
    
    def backward(self):
        """
        Computes backward pass of the loss

        Returns:
        np.ndarray 
             Gradient of loss wrt predicted output

        """
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[1]


class SGD:
    """
    Stochastic Gradient Desecent (SGD) Fucntion 
    """
    def __init__ (self,learning_rate):
        """
        Initializes the optimizer with a learning rate
        """
        self.learning_rate=learning_rate

    def step(self,layers):
        """
        Updates the weights and biases of the layers of the model

        Args:
        layers : list
             List of layers in the model
        """
        for layer in layers:
            if isinstance(layer, Linear):
                layer.weights -= self.learning_rate * layer.dweights
                layer.biases -= self.learning_rate * layer.dbiases

class Model:
    """
    A neural network model that allows for adding layers, compiling, training, evaluating, saving and loading.
    """
    def __init__(self):
        """
        Initializes the model with an empty list of layers, loss function and an optimizer        
        """
        self.layers=[]
        self.loss=None
        self.optimizer=None

    def add_layer(self,layer):
        """
        Adds a layer to the model

        Args:
        Layer : obejct
              A layer (Linear, ReLu, etc.) to add to the model
        """
        self.layers.append(layer)

    def compile(self,loss,optimizer):
        """
        Compiles the model with a loss function and an optimizer

        Args:
        loss : object
             Loss function to use
        
        optimizer : object
             Optimizer to use
        """
        self.loss=loss
        self.optimizer=optimizer

    def forward(self,X):
        """
        Performs forward pass through all the layers

        Args:
        X : np.ndarray
            Input data
        
        Returns:
        output : np.ndarray
             Output of the final layer
        """
        output=X
        for layer in self.layers:
            output=layer.forward(output)
        return output
    
    def backward(self):
        """
        Performs backward pass through all the layers
        """
        gradient=self.loss.backward()
        for layer in reversed(self.layers):
            gradient=layer.backward(gradient)
    
    def train(self,x_train,y_train,epochs,batch_size):
        """
        Trains the model
         
        Args:
        x_train : np.ndarray
               Training data

        y_train : np.ndarray
               Training labels
        
        epochs : int
               Number of epochs to train
        
        batch_size : int
               Batch size to use for training
        
               
        """
        
        for epoch in range(epochs):
           m=x_train.shape[1]
           
           for i in range(0,m,batch_size):
               x_batch=x_train[:,i:i+batch_size]
               y_batch=y_train[:,i:i+batch_size]

               
               y_pred=self.forward(x_batch)
               loss=self.loss.forward(y_pred,y_batch)
               self.backward()
               self.optimizer.step(self.layers)
              
       


    def predict(self,x_test):
        """
        Predicts the labels for test data

        Args:
        x_test : np.ndarray
               Test data
        
        Returns : np.ndarray
               Predicted labels
        """
        y_pred=self.forward(x_test)
        y_pred=np.argmax(y_pred,axis=0)
        return y_pred



    def evaluate(self,x_test,y_test):
        """
        Evaluates the performace of the model on test data

        Args:
        x_test : np.ndarray
               Test data
       
        y_test : np.ndarray
                Predicted Labels
        
        Returns :
        tuple :
               Loss and accuracy of model on test data     


        """
        y_pred=self.forward(x_test)
        loss=self.loss.forward(y_pred,y_test)
        y_pred=np.argmax(y_pred,axis=0)
        accuracy=100*np.sum((y_pred==y_test.flatten()))/y_test.shape[1]
        return loss,accuracy
    
    def save(self):
        """
        Saves the model parameters

        Returns:
        tuple
            The layers, loss, and optimizer of the model
        """
        return self.layers,self.loss,self.optimizer
    
    def load(self,saved_layers,saved_loss,saved_optimizer):
        """
        Loads the model parameters.

        Parameters:
        -----------
        saved_layers : list
                   Layers to load

        saved_loss : object
                   Loss function to load

        saved_optimizer : object
                   Optimizer to load
        
        """
        self.layers=saved_layers
        self.loss=saved_loss
        self.optimizer=saved_optimizer
