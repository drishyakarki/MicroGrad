from engine import Value
import random

class Module:
    def zero_grad(self): # Reset the grads of all parameters associated with the Module to zero
        for p in self.parameters():
            p.grad = 0

    def parameters(self): # returns list of parameters 
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # initialize weights randomly in range of -1 to 1
        self.b = Value(0) # bias to 0

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) # iterate over the tuple and calculate the weighted sum
        out = act.tanh() # applying tanh activation function
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # Creating neurons in layer with 'nin' input connections each
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # iterates through each neurons
        return outs[0] if len(outs) == 1 else outs # in output layer, there is only one neuron. so it directly returns the output
    
    def parameters(self):
        # gather all parameters from all neurns in layer and returns them as a single list
        return [p for neuron in self.neurons for p in neuron.parameters()] 
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # list with no. of i/p layers followed by no. of o/p neurons
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] 
        #MLP(3, [4, 4, 1]) --> 3 i/p neurons; 2 hidden layers with 4 neurons; 1 o/p layer with 1 neuron

    def __call__(self, x):
        # Iterate through each layer and pass x to each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # Gather all parameters from all layers in MLP and returns them as single list
        return [p for layer in self.layers for p in layer.parameters()]