Neural Network from Scratch
---
- the cost function is the loss function accumulated over the entire batch
- the activation of a neuron is the sum of all of it's predecessors, multiplied by each of their edge weights, and summed with their edge bias, all put into the activation function
- A weight exists for each edge between every neuron in two adjacent layers. A bias exists for each neuron in every succeeding layer. The input layer does not have a bias.
- The activation of a neuron is sum(weight_i * input_i) + bias. This explains why each neuron has a bias, and each edge has a weight.
- 
