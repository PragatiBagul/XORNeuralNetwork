import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ============================================================
# YOUR ORIGINAL DATASET FUNCTIONS (UNCHANGED)
# ============================================================

def generate_dataset(n_dim):
    initial_arr = [[0], [1]]

    for i in range(n_dim - 1):
        ans = []

        for ele in initial_arr:
            ans.append(ele + [0])
            ans.append(ele + [1])

        initial_arr = ans

    return initial_arr


def xor(arr, n):
    total_1 = sum(arr)

    if total_1 % 2 == 1:
        return 1

    return 0


def generate_true_labels(dataset):
    y = []

    for arr in dataset:
        ans = xor(arr, len(arr))
        y.append(ans)

    return y


# ============================================================
# YOUR ORIGINAL NEURAL NETWORK (MINIMAL CHANGES)
# ============================================================

class NeuralNetwork():

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights = {}
        self.bias = {}

        # Layer 1
        self.weights[0] = np.random.randn(hidden_dim, input_dim)
        self.bias[0] = np.random.randn(hidden_dim, 1).T

        # Layer 2
        self.weights[1] = np.random.randn(output_dim, hidden_dim)
        self.bias[1] = np.random.randn(output_dim, 1).T

        # Storing History
        self.history = {
            "loss": [],
            "w1": [],
            "w2": [],
            "predictions": []
        }

    def forward(self, x):
        self.input_ = x

        self.O_1 = self.sigmoid(
            (self.input_ @ self.weights[0].T) + self.bias[0]
        )

        self.y = self.sigmoid(
            (self.O_1 @ self.weights[1].T) + self.bias[1]
        )

        return self.y

    def compute_loss(self, y_true, y_pred):
        y_true = np.asarray(y_true)

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, lr, y, y_hat):

        m = len(y)

        # Output delta
        delta_2 = (y_hat - y) / m

        # Gradients layer 2
        dW2 = delta_2.T @ self.O_1
        db2 = np.sum(delta_2, axis=0, keepdims=True)

        # Hidden delta
        delta_1 = (
            delta_2 @ self.weights[1]
        ) * self.sigmoid_derivative(self.O_1)

        # Gradients layer 1
        dW1 = delta_1.T @ self.input_
        db1 = np.sum(delta_1, axis=0, keepdims=True)

        # UPDATE PARAMETERS
        self.weights[1] -= lr * dW2
        self.bias[1] -= lr * db2

        self.weights[0] -= lr * dW1
        self.bias[0] -= lr * db1

        return dW1, dW2


# ============================================================
# BATCH GRADIENT DESCENT
# ============================================================

def train_batch(epochs, X_train, y_train, nn, lr=0.1):

    for epoch in range(epochs):

        # Forward
        y_hat = nn.forward(X_train)

        # Loss
        loss = nn.compute_loss(y_train, y_hat)

        # Backward
        dW1, dW2 = nn.backward(lr, y_train, y_hat)

        # STORE HISTORY
        nn.history["loss"].append(loss)
        nn.history["w1"].append(nn.weights[0].copy())
        nn.history["w2"].append(nn.weights[1].copy())
        nn.history["predictions"].append(y_hat.copy())


# ============================================================
# SGD (YOUR ORIGINAL CODE WITH HISTORY ADDED)
# ============================================================

def train_sgd(epochs, X_train, y_train, nn, lr=0.1):

    m = X_train.shape[0]

    for epoch in range(epochs):

        indices = np.arange(m)
        np.random.shuffle(indices)

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0

        for i in range(m):

            x_sample = X_shuffled[i:i + 1]
            y_sample = y_shuffled[i:i + 1]

            y_hat = nn.forward(x_sample)

            epoch_loss += nn.compute_loss(y_sample, y_hat)

            nn.backward(lr, y_sample, y_hat)

        average_loss = epoch_loss / m

        # STORE HISTORY
        full_pred = nn.forward(X_train)

        nn.history["loss"].append(average_loss)
        nn.history["w1"].append(nn.weights[0].copy())
        nn.history["w2"].append(nn.weights[1].copy())
        nn.history["predictions"].append(full_pred.copy())


# ============================================================
# DATASET
# ============================================================

input_dim = 2

X_raw = generate_dataset(input_dim)
y_raw = generate_true_labels(X_raw)

X = np.array(X_raw, dtype=np.float32)
y = np.array(y_raw, dtype=np.float32).reshape(-1, 1)


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.title("XOR Neural Network Visualizer")

optimizer = st.sidebar.selectbox(
    "Optimizer",
    ["Batch Gradient Descent", "Stochastic Gradient Descent"]
)
input_dim = st.sidebar.slider("Hidden Neurons", 2, 100, 4)
hidden_dim = st.sidebar.slider("Hidden Neurons", 2, 10, 4)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    0.001,
    1.0,
    0.1
)

epochs = st.sidebar.slider(
    "Epochs",
    10,
    1000,
    200
)


# ============================================================
# TRAINING
# ============================================================

nn = NeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=1
)

if optimizer == "Batch Gradient Descent":
    train_batch(epochs, X, y, nn, learning_rate)
else:
    train_sgd(epochs, X, y, nn, learning_rate)


# ============================================================
# TITLE
# ============================================================

st.title("XOR Neural Network Visualization")

st.write(
    "Visualizing Weight Updates, Loss Reduction, and Gradient Descent"
)


# ============================================================
# LOSS CURVE
# ============================================================

st.subheader("Loss Curve")

fig_loss = go.Figure()

fig_loss.add_trace(
    go.Scatter(
        y=nn.history["loss"],
        mode='lines',
        name='Loss'
    )
)

fig_loss.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Loss"
)

st.plotly_chart(fig_loss, use_container_width=True)


# ============================================================
# WEIGHT VISUALIZATION
# ============================================================

st.subheader("Final Weight Matrices")

col1, col2 = st.columns(2)

with col1:
    st.write("Layer 1 Weights")
    st.dataframe(nn.weights[0])

with col2:
    st.write("Layer 2 Weights")
    st.dataframe(nn.weights[1])


# ============================================================
# PREDICTIONS
# ============================================================

st.subheader("Predictions")

final_pred = nn.forward(X)

prediction_table = {
    "Input": X.tolist(),
    "True Label": y.flatten(),
    "Prediction": final_pred.flatten()
}

st.dataframe(prediction_table)


# ============================================================
# DECISION BOUNDARY
# ============================================================

st.subheader("Decision Boundary")

xx, yy = np.meshgrid(
    np.linspace(-0.5, 1.5, 100),
    np.linspace(-0.5, 1.5, 100)
)

mesh_input = np.c_[xx.ravel(), yy.ravel()]

Z = nn.forward(mesh_input)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 6))

contour = ax.contourf(xx, yy, Z, alpha=0.7)

for i in range(len(X)):

    if y[i] == 0:
        ax.scatter(X[i][0], X[i][1], color='red', s=100)
    else:
        ax.scatter(X[i][0], X[i][1], color='blue', s=100)

ax.set_title("XOR Decision Boundary")

st.pyplot(fig)


# ============================================================
# EPOCH SCRUBBER
# ============================================================

st.subheader("Epoch History")

selected_epoch = st.slider(
    "Select Epoch",
    0,
    len(nn.history["loss"]) - 1,
    len(nn.history["loss"]) - 1
)

st.write(
    f"Loss at Epoch {selected_epoch}: ",
    nn.history["loss"][selected_epoch]
)

st.write("Weights Layer 1")
st.dataframe(nn.history["w1"][selected_epoch])

st.write("Weights Layer 2")
st.dataframe(nn.history["w2"][selected_epoch])


# ============================================================
# GRADIENT DESCENT EXPLANATION
# ============================================================

st.subheader("Gradient Descent Intuition")

if optimizer == "Batch Gradient Descent":
    st.success(
        "Batch GD computes gradients using the ENTIRE dataset before updating weights."
    )
else:
    st.warning(
        "SGD updates weights after EVERY sample, leading to noisy but faster learning."
    )


# ============================================================
# MATH SECTION
# ============================================================

st.subheader("Forward Propagation")

st.latex(r"Z_1 = XW_1 + b_1")
st.latex(r"A_1 = \sigma(Z_1)")
st.latex(r"Z_2 = A_1W_2 + b_2")
st.latex(r"\hat{y} = \sigma(Z_2)")

st.subheader("Backpropagation")

st.latex(r"W = W - \eta \frac{\partial L}{\partial W}")
