# task 13.3
## Notes
1. **Preceptons:** a type of artificial neurons that takes several binary inputs and produces a single binary output
   * The neuron's output, 0 or 1, is determined by whether the weighted sum ∑wx is less than or greater than some threshold value, where x is the binary input and w is its weight
   * Just like the weights, the threshold is a real number which is a parameter of the neuron
   * A precepton's bias is a measure of how easy it is to get the perceptron to output a  1, whereas for a perceptron with a really big bias, it's extremely easy to output a 1, but if the bias is very negative, then it's difficult for it to output a 1
   * We can use perceptrons to compute any logical function
     
2. **Sigmoid:** another type of artificial neuron similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output
   * Just like preceptons they have several binary inputs with their designated weights, but they can give off an output of any value between 0 and 1 unlike preceptons which are either 0 or 1
   * The output of a sigmoid is a sigmoid function which is:   **σ(z)≡ 1/(1 + e<sup>-z</sup>)**
   * the shape of the sigmoid function when plotted is a smoothed out version of a step function, while the shape of a percepton is in fact a step function, the smoothness of **σ** means that small changes **Δw<sub>j</sub>** in the weights and **Δb** in the bias will produce a small shift **Δoutput** in the output from the neuron
   * **Δoutput** is a linear function of the changes **Δw<sub>j</sub>** and **Δb** in the weights and bias. This linearity makes it easy to choose small changes in the weights and biases to achieve any desired small change in the output

3. **feedforward neural networks:** neural networks where the output from one layer is used as input to the next layer, which means there are no loops in the network - information is always fed forward, never fed back

4. We need to find weights and biases so that the output from the network approximates y(x) for all training inputs x, to quantify how well we're achieving this goal we define a **cost function**:   **C(w,b) ≡ (1/2n) ∑<sub>x</sub> ∥y(x) − a∥<sup>2</sup>**
   * w denotes the collection of all weights in the network
   * b all the biases
   * n is the total number of training inputs
   * a is the vector of outputs from the network when x is input
   *  and the sum is over all training inputs x
-> We aim to get to C(w,b)≈0, which means that our training algorithm has done a good job if it can find weights and biases, and on the contrary, it wouldn't be doing so well if C(w,b) is large, which would mean that y(x) is not close to the output a for a large number of inputs
  
5. **Gradient descent:** an optimization algorithm for finding a global minimum of a differentiable function, we use it to find the weights and biases which minimize the cost in the cost function, and so helping the net learn

6. **Stochastic gradient descent:** the idea behind it is to estimate the gradient ∇C by computing ∇C<sub>x</sub> for a small sample of randomly chosen training inputs, by averaging over this small sample we can quickly get a good estimate of the true gradient ∇C, which helps speed up gradient descent, and thus learning
   * stochastic gradient descent works by picking out a randomly chosen mini-batch of training inputs, and training with those, then we pick out another randomly chosen mini-batch and train with those. And so on, until we've exhausted the training inputs
     

## How It Works
### Classify each individual digit
  * To recognize individual digits a three-layer neural network is used, where:
    - The input layer contains 784=28×28 neurons, as the training data for the network consists of 28 by 28 pixel images of scanned handwritten digits
    - The second layer of the network is a hidden layer containing n neurons
    - The output layer of the network contains 10 neurons each corresponding to a digit from 0 to 9
      
### network.py
  * It's a module to implement the stochastic gradient descent learning algorithm for a feedforward neural network
  * The class `Network` is initialized with the parameter `sizes`, which is a list of integers specifying the number of neurons in each layer of the network, `self.num_layers` stores the number of layers in the network, in `self.biases` biases are initialized randomly for each neuron (except for input neurons), and in `self.weights` weights between the neurons of consecutive layers are also initialized randomly
    ```python
        def __init__(self, sizes):
          self.num_layers = len(sizes)
          self.sizes = sizes
          self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
          self.weights = [np.random.randn(y, x)
                          for x, y in zip(sizes[:-1], sizes[1:])]
    ```
  * The method `feedforward` takes an input `a` (the input to the network) and computes the output by propagating the signal through each layer, At each layer, it computes the weighted sum of inputs **(z = w * a + b)** and applies the sigmoid activation function, then the result becomes the input for the next layer
    ```python
        def feedforward(self, a):
          for b, w in zip(self.biases, self.weights):
              a = sigmoid(np.dot(w, a)+b)
          return a
    ```
  * The method `SGD` trains the model using the stochastic gradient descent algorithm which involves shuffling the training data, dividing the training data into smaller "mini-batches", updating the weights and biases using backpropagation for each mini-batch, evaluating the network's performance after each epoch if test data is provided
    ```python
        def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
          if test_data: n_test = len(test_data)
          n = len(training_data)
          for j in range(epochs):
              random.shuffle(training_data)
              mini_batches = [
                  training_data[k:k+mini_batch_size]
                  for k in range(0, n, mini_batch_size)]
              for mini_batch in mini_batches:
                  self.update_mini_batch(mini_batch, eta)
              if test_data:
                  print("Epoch {0}: {1} / {2}".format(
                      j, self.evaluate(test_data), n_test))
              else:
                  print("Epoch {0} complete".format(j))
    ```
  * The method `update_mini_batch` applies backpropagation to each mini-batch, calculates the gradient of the cost function with respect to the weights and biases, and updates them accordingly using the learning rate `eta`
    ```python
        def update_mini_batch(self, mini_batch, eta):
          nabla_b = [np.zeros(b.shape) for b in self.biases]
          nabla_w = [np.zeros(w.shape) for w in self.weights]
          for x, y in mini_batch:
              delta_nabla_b, delta_nabla_w = self.backprop(x, y)
              nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
              nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
          self.weights = [w-(eta/len(mini_batch))*nw
                          for w, nw in zip(self.weights, nabla_w)]
          self.biases = [b-(eta/len(mini_batch))*nb
                         for b, nb in zip(self.biases, nabla_b)]
    ```
  * The methode `backprop`  is the core algorithm for calculating the gradient of the cost function with respect to the weights and biases, consisting of two main steps:
      - **Feedforward:** It computes and stores all activations and weighted sums `z` at each layer as the input is passed through the network
      - **Backward Pass:** It computes the error `delta` at the output layer, then propagates this error backward through the network, adjusting the weights and biases layer by layer
      ```python
          def backprop(self, x, y):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)
      ```
  * The `evaluate` method checks how well the network performs on the test data by counting the number of correct predictions, and The network’s output is the index of the neuron in the final layer with the highest activation value
      ```python
          def evaluate(self, test_data):
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in test_data]
            return sum(int(x == y) for (x, y) in test_results)
      ```
  * The method `cost_derivative` computes the gradient of the cost function with respect to the output activations, which is necessary for calculating the error at the output layer during backpropagation
      ```python
          def cost_derivative(self, output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            return (output_activations-y)
      ```
  * The `sigmoid` method is used as the activation function in the neurons, as it outputs a value between 0 and 1 and the `sigmoid_prime` method is the derivative of the sigmoid function, used in backpropagation to calculate how much the activation function contributes to the error
      ```python
      def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    
      def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))
      ```
