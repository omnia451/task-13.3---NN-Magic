# task 13.3
## Notes
1. Preceptons: a type of artificial neurons that takes several binary inputs and produces a single binary output
   * The neuron's output, 0 or 1, is determined by whether the weighted sum ∑wx is less than or greater than some threshold value, where x is the binary input and w is its weight
   * Just like the weights, the threshold is a real number which is a parameter of the neuron
   * A precepton's bias is a measure of how easy it is to get the perceptron to output a  1, whereas for a perceptron with a really big bias, it's extremely easy to output a 1, but if the bias is very negative, then it's difficult for it to output a 1
   * We can use perceptrons to compute any logical function
2. Sigmoid: another type of artificial neuron similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output
   * Just like preceptons they have several binary inputs with their designated weights, but they can give off an output of any value between 0 and 1 unlike preceptons which are either 0 or 1
   * The output of a sigmoid is a sigmoid function which is:   **σ(z)≡ 1/(1 + e<sup>-z</sup>)**
   * the shape of the sigmoid function when plotted is a smoothed out version of a step function, while the shape of a percepton is in fact a step function, the smoothness of σ means that small changes Δw<sub>j</sub> in the weights and Δb in the bias will produce a small shift Δoutput in the output from the neuron
   * 
