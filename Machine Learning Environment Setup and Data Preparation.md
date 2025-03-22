# Machine Learning Lesson Contents: Neural Networks with PyTorch

## 1. Library Setup

**Objective:** Install and set up essential libraries within a virtual environment.

**Steps:**

1. **Create a Virtual Environment:**
   ```bash
   python -m venv ml_env
   ```

2. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     ml_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source ml_env/bin/activate
     ```

3. **Install Necessary Libraries:**
   ```bash
   pip install scikit-learn torch torchvision torchmetrics pandas seaborn matplotlib
   ```

## 2. Dataset Introduction

**Objective:** Load and explore the insurance charges dataset.

**Steps:**

1. **Load the Dataset:**
   ```python
   import pandas as pd

   df = pd.read_csv('insurance_charges.csv')
   ```

2. **Explore the Data Structure:**
   ```python
   print(df.head())
   print(df.info())
   ```

## 3. Data Preprocessing

**Objective:** Prepare the data for model training using encoding and scaling techniques.

**Steps:**

1. **Numerical Encoding of Categorical Variables:**
   ```python
   df = pd.get_dummies(df, columns=['smoker'], drop_first=True)
   ```

2. **Column Transformer:**
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler

   preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), ['age', 'bmi', 'children']),
           ('cat', OneHotEncoder(), ['sex', 'region'])
       ])
   ```

3. **Standard Scaling:**
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
   ```

4. **Min-Max Scaling:**
   ```python
   from sklearn.preprocessing import MinMaxScaler

   min_max_scaler = MinMaxScaler()
   df[['charges']] = min_max_scaler.fit_transform(df[['charges']])
   ```

5. **Convert Data to Torch Tensors:**
   ```python
   import torch

   X_tensor = torch.tensor(X.values, dtype=torch.float32)
   y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
   ```

## 4. Setting Up a Simple Neural Network

**Objective:** Build and train a simple neural network.

**Steps:**

1. **Define the Neural Network:**
   ```python
   import torch.nn as nn

   class SimpleNeuralNet(nn.Module):
       def __init__(self):
           super(SimpleNeuralNet, self).__init__()
           self.linear = nn.Linear(8, 1)

       def forward(self, x):
           return self.linear(x)
   ```

2. **Training the Neural Network:**
   ```python
   import torch.optim as optim
   import torch.nn.functional as F

   model = SimpleNeuralNet()
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   loss_fn = nn.MSELoss()

   # Training loop
   for epoch in range(100):
       model.train()
       optimizer.zero_grad()
       outputs = model(X_tensor)
       loss = loss_fn(outputs, y_tensor)
       loss.backward()
       optimizer.step()
   ```

## 5. Dataset and DataLoader

**Objective:** Prepare data for efficient training.

**Steps:**

1. **Create a TensorDataset:**
   ```python
   from torch.utils.data import TensorDataset

   dataset = TensorDataset(X_tensor, y_tensor)
   ```

2. **Set Up DataLoader:**
   ```python
   from torch.utils.data import DataLoader

   dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
   ```

## 6. Training Loop

**Objective:** Train the neural network and track performance.

**Steps:**

1. **Epochs and Loss Stats:**
   ```python
   loss_stats = {'train': [], 'val': []}
   ```

2. **Device Selection:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

3. **Model Training:**
   ```python
   for epoch in range(100):
       model.train()
       for batch in dataloader:
           X_batch, y_batch = batch
           X_batch, y_batch = X_batch.to(device), y_batch.to(device)
           optimizer.zero_grad()
           outputs = model(X_batch)
           loss = loss_fn(outputs, y_batch)
           loss.backward()
           optimizer.step()
           loss_stats['train'].append(loss.item())
   ```

4. **Validation:**
   ```python
   model.eval()
   with torch.no_grad():
       val_outputs = model(X_tensor)
       val_loss = loss_fn(val_outputs, y_tensor)
       loss_stats['val'].append(val_loss.item())
   ```

## 7. Visualizing Losses

**Objective:** Monitor training and validation losses.

**Steps:**

1. **Plot Losses:**
   ```python
   import matplotlib.pyplot as plt

   plt.plot(loss_stats['train'], label='Training Loss')
   plt.plot(loss_stats['val'], label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()
   ```

## 8. Evaluating Models

**Objective:** Assess model performance using metrics.

**Steps:**

1. **R-square Score:**
   ```python
   from sklearn.metrics import r2_score

   r2 = r2_score(y_tensor.cpu(), val_outputs.cpu())
   print(f'R-square Score: {r2}')
   ```

## 9. Key Concepts

**Objective:** Understand the differences between simple and complex neural networks.

**Steps:**

1. **Simple vs. Complex Neural Network:**
   - **Simple Neural Network:** Single neuron, linear regression-like.
   - **Complex Neural Network:** Multiple layers and neurons, learns intricate patterns.

2. **Layers and Neurons:**
   - **Layer 1:** 16 neurons, input features.
   - **Layer 2:** 32 neurons.
   - **Layer 3:** 16 neurons.
   - **Output Layer:** 1 neuron, final prediction.

3. **Activation Function (ReLU):**
   ```python
   import torch.nn.functional as F

   class ComplexNeuralNet(nn.Module):
       def __init__(self):
           super(ComplexNeuralNet, self).__init__()
           self.layer1 = nn.Linear(8, 16)
           self.layer2 = nn.Linear(16, 32)
           self.layer3 = nn.Linear(32, 16)
           self.output = nn.Linear(16, 1)

       def forward(self, x):
           x = F.relu(self.layer1(x))
           x = F.relu(self.layer2(x))
           x = F.relu(self.layer3(x))
           return self.output(x)
   ```

4. **Training Process:**
   - Initialize weights, set up DataLoader, and train for 100 epochs.

5. **Performance Metrics:**
   - Track training and validation losses and calculate the R-square score.

## Summary

This lesson covers the essential steps for setting up a machine learning environment, preprocessing data, building a neural network, and evaluating its performance using PyTorch. By following these steps, you can effectively prepare and train a regression model on an insurance charges dataset.

---

**Note:** Ensure you have the `insurance_charges.csv` dataset available in your working directory to follow along with the code examples.
