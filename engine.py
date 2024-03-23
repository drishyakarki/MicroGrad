import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0 
        self._op = _op  # the operator 
        self._backward = lambda: None # eg: for leaf node there is nothing to do. so, it is empty 
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})" # -----> Value(data=4.0, grad=0.4)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # handling cases like 2 + a
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad # derivative of a + b wrt to a is simply 1 and we simply pass the global grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out 
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad # derivative of say a*b wrt a is simply b
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "supports only int/float"
        out = Value(self.data**other, (self, ), f'**{other}') 

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad # derivative of power
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad # derivative of tanh is 1 -tanh ^ 2
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward(self):
            self.grad += out.grad * out.data
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1 # a /b = a * b ^ -1
    
    def __rtruediv__(self, other):
        return other * self**-1 # other / self
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topo = [] # initialize empty list topo to store topological ordering of the nodes in the computation graph
        visited = set()
        # Recursively performs depth-first search(DFS) traversal of the computation graph
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo): # start from the output 
            node._backward() # calls backward function associated with each node in the reversed topological order