
# My Learning Notes
    
- **Building Blocks of Neural Networks**: By working with Micrograd, users can gain insights into the fundamental building blocks of neural networks, including forward and backward passes.
    - **Forward Pass**
        
        The forward pass refers to the process of feeding input data through the network to obtain the output (predictions). This step involves calculating the activations of each layer sequentially, starting from the input layer and moving through to the output layer. 
        
    - **Backward pass**
        
        The backward pass, or backpropagation, is the process of updating the network's weights and biases based on the loss function's gradients. This step involves calculating the gradients of the loss with respect to each weight and bias in the network and using these gradients to update the parameters.

    - **What is gradient decent:**

    Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models, including neural networks. It iteratively adjusts the model's parameters (weights and biases) to reduce the difference between the predicted output and the actual output.

### Gradient Descent in Neural Networks

In neural networks, the gradient descent algorithm involves the following steps:

1. **Forward Pass**: Compute the output of the network by passing the input data through each layer.
2. **Compute Loss**: Calculate the loss using a loss function (e.g., mean squared error, cross-entropy).
3. **Backward Pass**: Compute the gradients of the loss with respect to each parameter using backpropagation.
4. **Parameter Update**: Update the parameters using the gradient descent update rule.

### Activation Functions:
- Tanh: smoother, more complicatated, hence, stressed more on the gradient
- Softmax: it converts the scores to a normalized probability distribution, which can be displayed to a user or used as input to other systems. For this reason it is usual to append a softmax function as the final layer of the neural network.
- Relu

### Interesting point: 
- learned the most NN mistakes: forgot to .zero_grad() before .backward() in pytorch

# From Andrej

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
