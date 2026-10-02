import numpy as np
data=np.load("Data\extend_mnist_eval.npz")
print(data.files)
x=data["img"]
y=data["label"]
print("X shape :: ", x.shape)
print("Y shape :: ", y.shape)
print("X datatype :: ", x.dtype)
print("Y datatype :: ", y.dtype)
print("X min :: ", x.min())
print("Y min :: ", y.min())
print("X max :: ", x.max())
print("Y max :: ", y.max())
print("First label:", y[0])
import matplotlib.pyplot as plt

plt.imshow(x[0], cmap="gray")
plt.title(f"Label :: {int(y[0])}")
plt.axis("off")
plt.show()
x=x/255.0
print(x.min())
print(x.max())
x=x.reshape(x.shape[0],-1)
x.shape
y=y.astype(int)
print(y.dtype)
print(np.unique(y))
classes,count=np.unique(y, return_counts=True)
for c,count in zip(classes, count):
    print(c, count)
ineurons=128
w1=np.random.randn(x.shape[1], ineurons)
b1=np.zeros((1, ineurons))
oneurons=16
w2=np.random.randn(ineurons, oneurons)
b2=np.zeros((1, oneurons))
print(w1.shape)
print(b1.shape)
print(w2.shape)
print(b2.shape)
z1=np.dot(x,w1)+b1
print(z1.shape)
def relu(x):
    return np.maximum(0,x)
A1 = relu(z1)
A1.shape
z2=np.dot(A1, w2)+b2
def softmax(x):
    exp_x=np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)
A2=softmax(z2)
print(A2.shape)
print(A2[0])
np.argmax(A2[0])
prediction = np.argmax(A2[0])
correct_prediction=y[0]
print("Prediction:", prediction)
print("Actual:", y[0])
correct_class = y[0]
correct_probability = A2[0, correct_class]

epsilon = 1e-8
loss = -np.log(correct_probability + epsilon)

print("Actual class:", correct_class)
print("Probability:", correct_probability)
print("Loss:", loss)
indices = np.arange(len(y))
correct_probabilities = A2[indices, y]
epsilon = 1e-8
losses = -np.log(correct_probabilities + epsilon)
loss = np.mean(losses)
print("Loss:", loss)

