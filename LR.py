#=====Without mini-batch=====#
# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import fetch_openml

# # -------------------------------
# # Load MNIST
# # -------------------------------
# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# X = mnist.data.astype("float32") / 255.0
# y = mnist.target.astype(int)

# train_X, test_X = X[:60000], X[60000:]
# train_y, test_y = y[:60000], y[60000:]

# # -------------------------------
# # One-hot encode labels
# # -------------------------------
# def one_hot(labels, num_classes):
#     oh = np.zeros((labels.size, num_classes))
#     oh[np.arange(labels.size), labels] = 1
#     return oh

# train_y_oh = one_hot(train_y, 10)

# # -------------------------------
# # Initialize Parameters
# # -------------------------------
# W = np.zeros((784, 10))
# b = np.zeros((1, 10))

# # -------------------------------
# # Softmax
# # -------------------------------
# def softmax(z):
#     exp = np.exp(z - np.max(z, axis=1, keepdims=True))
#     return exp / np.sum(exp, axis=1, keepdims=True)

# # -------------------------------
# # Logistic Regression Training (full batch)
# # -------------------------------
# def train_full_batch(X, y, lr=0.1, epochs=10):
#     global W, b
#     n = X.shape[0]

#     for epoch in range(epochs):
#         logits = X @ W + b #multiply inputs by weights and add bias
#         probs = softmax(logits)

#         grad_W = X.T @ (probs - y) / n
#         grad_b = np.sum(probs - y, axis=0, keepdims=True) / n

#         W -= lr * grad_W
#         b -= lr * grad_b

#         print(f"Epoch {epoch+1} completed")


# train_full_batch(train_X, train_y_oh, lr=0.1, epochs=10)

# # -------------------------------
# # Test
# # -------------------------------
# logits = test_X @ W + b
# probs = softmax(logits)
# preds = np.argmax(probs, axis=1)

# print("Accuracy:", accuracy_score(test_y, preds))

#With
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# -------------------------------
# Load MNIST
# -------------------------------
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype(int)

train_X, test_X = X[:60000], X[60000:]
train_y, test_y = y[:60000], y[60000:]

# -------------------------------
# One-hot encode labels
# -------------------------------
def one_hot(labels, num_classes):
    oh = np.zeros((labels.size, num_classes))
    oh[np.arange(labels.size), labels] = 1
    return oh

train_y_oh = one_hot(train_y, 10)

# -------------------------------
# Initialize parameters
# -------------------------------
W = np.zeros((784, 10))
b = np.zeros((1, 10))

# -------------------------------
# Softmax
# -------------------------------
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# -------------------------------
# MINI-BATCH TRAINING
# -------------------------------
def train_minibatch(X, y, batch_size=128, lr=0.1, epochs=10):
    global W, b
    n = X.shape[0]

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            logits = X_batch @ W + b
            probs = softmax(logits)

            grad_W = X_batch.T @ (probs - y_batch) / batch_size
            grad_b = np.sum(probs - y_batch, axis=0, keepdims=True) / batch_size

            W -= lr * grad_W
            b -= lr * grad_b

        print(f"Epoch {epoch+1} completed")

train_X = train_X.to_numpy() if hasattr(train_X, "to_numpy") else train_X
train_y_oh = train_y_oh.to_numpy() if hasattr(train_y_oh, "to_numpy") else train_y_oh
test_X = test_X.to_numpy() if hasattr(test_X, "to_numpy") else test_X
train_minibatch(train_X, train_y_oh, batch_size=128, lr=0.1, epochs=10)

# -------------------------------
# Test
# -------------------------------
logits = test_X @ W + b
probs = softmax(logits)
preds = np.argmax(probs, axis=1)

print("Accuracy:", accuracy_score(test_y, preds))
