#train a deep learning model to predict whether a game will be successful or not. A game is successful if it has more than 500000 owners.
#predict sucess of a game (owners > 500000) based on features. feature in dataset are ['appid', 'name', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'owners', 'price']

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

class GameSuccessPredictor(nn.Module):
    def __init__(self, input_dim):
        super(GameSuccessPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_and_save_model():
    ### Step 2: Load and Preprocess Data

    # Load the dataset
    df = pd.read_csv('./datasets/descMediaReqSupp/steam.csv')

    # Fill missing values
    df.fillna('', inplace=True)

    # Convert release_date to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Extract year from release_date
    df['release_year'] = df['release_date'].dt.year

    # Handle owners column by taking the midpoint of the range
    df['owners'] = df['owners'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) // 2)

    # Encode categorical features
    label_encoders = {}
    for column in ['developer', 'publisher', 'platforms', 'categories', 'genres', 'steamspy_tags']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define the target variable (success metric)
    df['success'] = (df['positive_ratings'] > df['negative_ratings']).astype(int)

    # Select features and target
    features = ['english', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'achievements', 'owners', 'price', 'release_year']
    target = 'success'

    X = df[features].values
    y = df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ### Step 3: Create DataLoader

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ### Step 4: Define the Model

    input_dim = X_train.shape[1]
    model = GameSuccessPredictor(input_dim)

    ### Step 5: Train the Model

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Save the model
    torch.save(model.state_dict(), 'DLgameSuccess.pth')
    print('Model saved to DLgameSuccess.pth')

    ### Step 6: Evaluate the Model

    # Evaluate the Model with additional metrics
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate accuracy
    accuracy = (all_predictions == all_labels).mean()

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

if __name__ == "__main__":
    train_and_save_model()
