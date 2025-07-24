import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        # Config per discretizzazione
        self.config = config
        if config:
            self.weight_values = np.round(
                np.arange(config["weight_min"], config["weight_max"] + config["weight_step"], config["weight_step"]),
                4,
            )
        else:
            self.weight_values = None  # fallback

    def get_total_weights(self):
        w1_size = self.input_size * self.hidden_size
        b1_size = self.hidden_size
        w2_size = self.hidden_size * self.output_size
        b2_size = self.output_size
        return w1_size + b1_size + w2_size + b2_size

    def forward_with_weights(self, x, weights_vector):
        w1_size = self.input_size * self.hidden_size
        b1_size = self.hidden_size
        w2_size = self.hidden_size * self.output_size

        w1 = weights_vector[:w1_size].reshape(self.hidden_size, self.input_size)
        b1 = weights_vector[w1_size : w1_size + b1_size]
        w2 = weights_vector[w1_size + b1_size : w1_size + b1_size + w2_size].reshape(
            self.output_size, self.hidden_size
        )
        b2 = weights_vector[w1_size + b1_size + w2_size :]

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if not isinstance(w1, torch.Tensor):
            w1 = torch.from_numpy(w1).float()
        if not isinstance(b1, torch.Tensor):
            b1 = torch.from_numpy(b1).float()
        if not isinstance(w2, torch.Tensor):
            w2 = torch.from_numpy(w2).float()
        if not isinstance(b2, torch.Tensor):
            b2 = torch.from_numpy(b2).float()
        if not isinstance(weights_vector, torch.Tensor):
            weights_vector = torch.from_numpy(weights_vector).float()

        hidden_output = self.activation(torch.matmul(x, w1.T) + b1)
        final_output = torch.matmul(hidden_output, w2.T) + b2

        return final_output

    # --- Metodo per ottenere pesi discreti ---
    def get_discrete_weights(self):
        if self.weight_values is None:
            raise ValueError("weight_values non definito, assicurarsi che 'config' sia passato al costruttore MLP")
        total_weights = self.get_total_weights()
        return np.random.choice(self.weight_values, size=total_weights)

    # --- Heuristica (MSE) ---
    def heuristic(self, X_train, y_train_ohe_tensor, weights_vector):
        with torch.no_grad():
            predictions = self.forward_with_weights(X_train, weights_vector)
            loss = nn.MSELoss()
            mse = loss(predictions, y_train_ohe_tensor).item()
        return mse
