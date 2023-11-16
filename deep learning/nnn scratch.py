import numpy as np
import pandas as pd

# Load your dataset (make sure to replace 'heart_disease_dataset.csv' with your dataset's filename)
df = pd.read_csv("dataset.csv")
df.head()

# Extract features (x) and target (y)
x = df.drop('target', axis=1)
y = df['target']
x = np.array(x)
y = np.array(y)

# Shuffle the data
index = np.random.permutation(len(df))
x = x[index]
y = y[index]

# Define batch size
batch_size = 16

# Initialize lists to store mini-batches
mini_batches_x = []
batches_y = []

# Create mini-batches
for batch_idx in range(len(df) // batch_size):
    start_point = batch_idx * batch_size
    end_point = (batch_idx + 1) * batch_size
    mini_batch = x[start_point:end_point]
    mini_batch_target = y[start_point:end_point]
    mini_batches_x.append(mini_batch)
    batches_y.append(mini_batch_target)

if len(df) % batch_size != 0:
    start_idx = (len(df) // batch_size) * batch_size
    mini_batch_x = x[start_idx:]
    mini_batch_y = y[start_idx:]
    mini_batches_x.append(mini_batch_x)
    batches_y.append(mini_batch_y)

# Learning rate
lr = 0.01

# Random seed for reproducibility
np.random.seed(0)

# Define the neural network architecture (adjust the sizes as needed)
input_size = len(df.columns) - 1
hidden_sizes = [40, 6, 4]
output_size = 1

# Initialize weights and biases for each layer
weights = [np.random.randn(input_size, hidden_sizes[0])]
weights += [np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)]
weights += [np.random.randn(hidden_sizes[-1], output_size)]

biases = [np.zeros((1, size)) for size in hidden_sizes]
biases += [np.zeros((1, output_size))]

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Loss function
def loss(y_pred, y_real):
    return - (y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

# Number of training epochs
num_epochs = 1000

# Training the neural network
for epoch in range(num_epochs):
    average_loss = 0

    for i in range(len(mini_batches_x)):
        input_data = mini_batches_x[i]
        target = batches_y[i]

        # Lists to store intermediate outputs and deltas
        layer_outputs = []
        layer_deltas = []

        # Forward pass
        layer_output = input_data
        for j in range(len(hidden_sizes) + 1):
            layer_inputs = np.dot(layer_output, weights[j]) + biases[j]
            layer_output = sigmoid(layer_inputs)
            layer_outputs.append(layer_output)

        # Calculate loss
        batch_loss = np.mean(loss(layer_output, target))
        average_loss += batch_loss

        # Backpropagation
        output_error = target - layer_output
        output_delta = output_error * sigmoid_derivative(layer_output)
        layer_deltas.append(output_delta)

        for j in range(len(hidden_sizes), 0, -1):
            hidden_error = layer_deltas[j].dot(weights[j].T)  # Corrected line
            hidden_delta = hidden_error * sigmoid_derivative(layer_outputs[j])
            layer_deltas.insert(0, hidden_delta)

        # Update weights and biases
        for j in range(len(hidden_sizes) + 1):
            weights[j] += layer_outputs[j].T.dot(layer_deltas[j]) * lr
            biases[j] += np.sum(layer_deltas[j], axis=0, keepdims=True) * lr

    average_loss /= len(mini_batches_x)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    