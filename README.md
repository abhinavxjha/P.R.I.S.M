<div align="center">

# 🔺 P.R.I.S.M

### *Teaching a neural network to read the language of mathematics, one handwritten stroke at a time.*

![Status](https://img.shields.io/badge/status-work_in_progress-orange?style=for-the-badge)
![Type](https://img.shields.io/badge/neural_network-built_from_scratch-blueviolet?style=for-the-badge)
![Input](https://img.shields.io/badge/input-28×28_images-00c2a8?style=for-the-badge)
![Made at](https://img.shields.io/badge/made_at-JUIT_Solan-red?style=for-the-badge)

**`✍️ handwritten image`** ➜ **`🧠 neural network`** ➜ **`🔢 digit · ➕ operator · 🔲 bracket`**

</div>

---

## 👁️ What is PRISM?

A prism takes one beam of light and splits it into something you can actually read.

**PRISM does the same for handwriting.** It takes a messy, human-drawn 28×28 pixel image and splits it into a clear answer: *is this a digit, a mathematical operator, or a bracket?*

No black-box one-liners. The goal of this project is to **understand every gear inside a neural network**, from raw pixels to weight updates, by building it piece by piece instead of calling `model.fit()`.

---

## 🎯 The Mission

| Input | Output |
|-------|--------|
| A handwritten 28×28 grayscale image | A predicted class label |

**Target classes**

- 🔢 **Digits**: `0 1 2 3 4 5 6 7 8 9`
- ➕ **Operators**: `+ − × ÷ =` and friends
- 🔲 **Brackets**: `( )` and friends

*(Update this list to match your exact dataset classes.)*

---

## 🧬 How It Works

```
   ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
   │  28 × 28   │     │   Hidden   │     │   Hidden   │     │   Output   │
   │   image    │ ──▶ │   Layer    │ ──▶ │ activation │ ──▶ │  softmax   │ ──▶  prediction
   │ (784 px)   │     │  W·x + b   │     │            │     │ probabilities│
   └────────────┘     └────────────┘     └────────────┘     └────────────┘
         ▲                                                          │
         │                 ◀── backpropagation ◀── loss ◀───────────┘
         └──────────── gradient descent updates weights & biases
```

*(Adjust the diagram to match your actual layer count and activations.)*

**The learning loop, in plain English:**

1. **Forward pass**: pixels flow through the layers and the network makes a guess.
2. **Loss**: measure *how wrong* that guess was.
3. **Backpropagation**: trace the blame backward through every layer.
4. **Gradient descent**: nudge every weight and bias a tiny step toward "less wrong".
5. **Repeat** for thousands of images, over many epochs, until the guesses get good.

---

## 🚧 Current Status

> **PRISM is under active construction.** The foundation is in place, and the learning machinery is the next big milestone.

| Stage | Status |
|-------|:------:|
| Dataset loading & preprocessing | ✅ |
| Network structure & forward pass | ✅ |
| One-hot encoding | 🔜 |
| Loss + backpropagation | 🔜 |
| Gradient calculation & gradient descent | 🔜 |
| Training loop with metrics | 🔜 |
| Evaluation on test set | 🔜 |
| Visualizations | 🔜 |
| Hyperparameter experiments | 🔜 |
| Save / load model + prediction pipeline | 🔜 |

---

## 🗺️ Roadmap: Future Development

The next stage turns PRISM from a network that *computes* into a network that *learns*.

### ⚙️ Phase 1: Make it learn
- [ ] **One-hot encoding** of labels so the network can compare predictions against targets
- [ ] **Backpropagation** to push error backward through every layer
- [ ] **Gradient calculation** for all weights and biases
- [ ] **Gradient descent** so the network updates itself from its mistakes

### 🏋️ Phase 2: Train it
- [ ] Train over **multiple epochs**
- [ ] Track **loss and accuracy** every epoch
- [ ] **Evaluate on the held-out test dataset**
- [ ] **Visualize** the training curves and sample predictions

### 🔬 Phase 3: Understand it
- [ ] Experiment with different **learning rates**
- [ ] Experiment with different **hidden-layer sizes**
- [ ] Tune other **hyperparameters** and study how each one changes performance

### 🚀 Phase 4: Ship it
- [ ] **Save the trained parameters** (weights and biases)
- [ ] Build a **prediction pipeline**: feed in any new 28×28 handwritten image and get back a **digit, operator, or bracket**

---

## 🧮 The Math Behind the Magic

**Forward pass**

$$z = W x + b \qquad a = f(z)$$

**Loss** *(how wrong are we?)*

$$L = -\sum_i y_i \log(\hat{y}_i)$$

**Gradient descent update** *(how we get better)*

$$W \leftarrow W - \eta \frac{\partial L}{\partial W} \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

where **η** is the learning rate, one of the hyperparameters this project will explore.

---

## 🛠️ Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/abhinavxjha/P.R.I.S.M.git
cd P.R.I.S.M

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook / run the project
jupyter notebook
```

*(Edit these steps to match your actual file names and setup.)*

---

## 🔮 Planned Usage

Once the prediction pipeline lands, using PRISM will look like this:

```python
from prism import load_model, predict

model = load_model("prism_weights")
label = predict(model, "my_handwritten_symbol.png")

print(label)   # → "+"   (or "7", or "(" ...)
```

*(Illustrative only. The final API will be documented when it exists.)*

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

*(Adjust badges to the libraries you actually use.)*

---

## 💡 Why Build It From Scratch?

Because anyone can import a framework. Writing backpropagation by hand is how you *really* learn what a gradient is, why learning rates blow up, and what a hidden layer is actually doing. PRISM is as much a learning journey as it is a classifier.

---

## 🤝 Contributing

Ideas, bug reports, and suggestions are welcome. Open an issue or a pull request.

---

## 👤 Author

**Abhinav Jha**
B.Tech CSE, Jaypee University of Information Technology (JUIT), Solan
🔗 [GitHub: @abhinavxjha](https://github.com/abhinavxjha)

---

<div align="center">

⭐ **If you like watching a neural network learn to read, star the repo!** ⭐

*Light in. Meaning out.*

</div>
