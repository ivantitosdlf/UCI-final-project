import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import copy
import joblib

from model import HeartDiseaseNN

# ================= CONFIG =================
dataset_path = 'heart.csv'
epochs = 250

# ================ FUNCTIONS ===============

def data_preparation(dataset_path):
    # Load and preprocess data
    data = pd.read_csv(dataset_path)

    # Encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Split into features (X) and target (y)
    X = data.drop('HeartDisease', axis=1).values
    y = data['HeartDisease'].values

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler
    joblib.dump(scaler, 'output/scalers/scaler.pkl')
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # Create DataLoaders
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    return train_loader, val_loader, X_train.shape[1]


def train_NN(model, epochs, train_loader, val_loader, patience=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Training setup
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # For early stopping and model saving
    best_val_acc = 0.0
    best_model = None
    best_epoch = 0
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Print metrics
        print(f'Epoch {epoch+1}/{epochs} -- '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} -- '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} '
              f'{"(Best)" if val_acc == best_val_acc else ""}')
        
        # Early stopping check
        if patience and epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            print(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')
            break
    
    # Load best model weights
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/plots/train_validation_plot.png')
    print('Saved train validation plot at: output/plots ')
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')

    torch.save(model.state_dict(), 'output/trained_models/trained_HeartDiseaseNN.pth')
    print('Saved model at output/trained_models.')
    
    return model


# ============== MAIN ===============
def main():
    train_loader, val_loader, input_size = data_preparation(dataset_path)
    model = HeartDiseaseNN(input_size)
    train_NN(model = model, 
             epochs=epochs,
             train_loader=train_loader,
             val_loader=val_loader)
    
if __name__ == '__main__':
    main()
    print('End of program')