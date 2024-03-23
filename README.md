# Micrograd

Micrograd is a minimalistic deep learning library implemented in python, providing automatic differentiation capabilities for building and training neural networks. Inspired by the teachings of [Andrej Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1), this project aims to provide a simple yet educational implementation of automatic differentiation, a crucial component of modern deep learning frameworks.

## Introduction

The code in this repository implements a basic autograd engine capable of computing gradients for functions involving scalar operations, such as addition, multiplication, exponentiation, and hyperbolic tangent (tanh) activation. The engine supports both forward and backward passes, allowing users to compute gradients efficiently through the computational graph.

The core of the autograd engine is the Value class, which represents scalar values and tracks their gradients during computation. By overloading arithmetic operators such as addition, multiplication, and exponentiation, the Value class enables the construction of complex computational graphs with automatic gradient computation.

To demonstrate the functionality of the autograd engine, the repository includes test script [run_tests.py](run_tests.py) that compare the gradients computed by Micrograd with those computed by PyTorch, a widely used deep learning framework. These tests validate the correctness of the gradients computed by Micrograd and serve as educational examples for understanding automatic differentiation.

Additionally, the repository includes a simple neural network module (nn.py) built on top of Micrograd. The neural network module implements a multi-layer perceptron (MLP) architecture, consisting of layers of neurons with trainable weights and biases. Users can create and train neural networks using the provided module, gaining hands-on experience with automatic differentiation and neural network training.

## Example Usage

Below is a simpple example demonstrating micrograd supported operations:
```python
from engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()
d += 3 * d + (b - a).tanh()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # 4.6252
g.backward()
print(f'{a.grad:.4f}') # 27.0601
print(f'{b.grad:.4f}') # 117.3345
```

### Classes and methods avaiable in the library

- `Value`: Represents a node in the computation graph with automatic differentiation capabilities.
- `Module`: Base class for neural network modules.
- `Neuron`: Represents a single neuron in a neural network.
- `Layer`: Represents a layer of neurons in a neural network.
- `MLP`: Represents a multi-layer perceptron (neural network) composed of multiple layers.

## To run unit tests

Clone the repo. Then, you will have to install PyTorch and Numpy library.You can directly run ```pip install numpy torch```. 

Then simply run: 

```bash
python run_tests.py
```