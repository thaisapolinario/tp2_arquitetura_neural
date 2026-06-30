import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


#  Modelos 

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])


#  Dataset 

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i: i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)


#  Treinamento 

def treinar(model, X_train, y_train, epochs=30, batch_size=32, lr=0.001, verbose=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    historico_loss = []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = torch.FloatTensor(y_train[i:i + batch_size])

            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        media = epoch_loss / n_batches
        historico_loss.append(media)
        if verbose:
            print(f"  Época [{epoch + 1:02d}/{epochs}]  Loss: {media:.6f}")

    return historico_loss


#  Previsão 

def prever(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
    pred_real = scaler.inverse_transform(pred)
    y_test_real = scaler.inverse_transform(y_test)
    return pred_real, y_test_real


#  Métricas 

def calcular_metricas(y_real, y_pred, nome):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    print(f"--- Métricas {nome} ---")
    print(f"  RMSE : R$ {rmse:.4f}")
    print(f"  MAE  : R$ {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return rmse, mae, mape


#  Contagem de parâmetros 

def contar_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)