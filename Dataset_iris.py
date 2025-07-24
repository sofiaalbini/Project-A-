import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Dataset:
    def __init__(self, batch_size=16):
        # Caricamento dataset Iris
        iris = load_iris()
        X, y = iris.data, iris.target

        # Suddivisione in train/test con stratificazione
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardizzazione delle feature
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # One-Hot Encoding delle etichette di training
        ohe = OneHotEncoder(sparse_output=False)
        y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

        # Conversione in tensori PyTorch
        X_train_tensor = torch.from_numpy(X_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_train_ohe_tensor = torch.from_numpy(y_train_ohe).float()
        y_train_labels_tensor = torch.from_numpy(y_train).long()
        y_test_labels_tensor = torch.from_numpy(y_test).long()

        # Creazione dataset e dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_ohe_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  
        self.X_test = X_test_tensor
        self.y_test_labels = y_test_labels_tensor
        self.y_train_labels = y_train_labels_tensor  # Etichette originali (non one-hot)
        self.y_train_ohe_tensor = y_train_ohe
