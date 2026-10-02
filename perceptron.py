# A perceptron mimics a biological neuron by processing data in three clear steps:Weighted Sum: It takes multiple numeric inputs (x₁, x₂, ...), multiplies each by a specific weight (w₁, w₂, ...), and adds them together along with a threshold-shifting value called a bias (b). This creates a single value: z = (x₁w₁ + x₂w₂ + ...) + b.Activation: It passes this total (z) through a hard-threshold activation function (a step function). If z is greater than or equal to 0, the perceptron outputs a 1. If z is less than 0, it outputs a 0.Learning: If the output is incorrect during training, the perceptron adjusts its weights and bias slightly in the direction of the correct answer.The Core LimitationA single perceptron is a linear classifier. This means it can only successfully solve problems where the data can be perfectly split by a straight line (known as linearly separable data, like the logical AND or OR gates). It completely fails at problems that require a curved or complex split, like the logical XOR gate. To solve complex problems, multiple perceptrons must be stacked together to create a Deep Neural Network.

def step(x):
    return 1 if x>=0 else 0

def perceptron(x1, x2, w1, w2, b):
    z = x1*w1 + x2*w2 + b
    return step(z)

print(perceptron(0, 0, 1, 1,-1.5)) 
print(perceptron(0, 1, 1, 1,-1.5)) 
print(perceptron(1, 0, 1, 1,-1.5)) 
print(perceptron(1, 1, 1, 1,-1.5)) 


#this project will acts as an AND gate, the output is 1 only when both inputs are 1. The weights and bias are set to achieve this behavior.


