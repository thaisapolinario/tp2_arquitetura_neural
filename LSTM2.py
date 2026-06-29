import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------
#  Dados da Petrobrás de 2016 a 2026
# -----------------------------------
df = yf.download("PETR4.SA", start="2016-01-01", end="2026-01-01")
df = df[['Close']]

# --------------------------
#  Normalizar dados (0 a 1)
# --------------------------
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# ----------------
#  Criar janelas
# ----------------
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i: i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(df_scaled)

# ------------------------
#  Separar treino e teste
# ------------------------
split = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split])
X_test  = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split])
y_test  = y[split:]   # Mantido em numpy para métricas finais

# ------------------------------
#  Criar modelo LSTM em PyTorch
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        # LSTM: possui 3 gates (input, forget, output) e cell state
        # Mais parâmetros que a GRU, porém mais expressiva em sequências longas
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Último passo temporal
        return out

model = LSTMModel(hidden_size=50, num_layers=2, dropout=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------
#  Treinar
# ----------
epochs     = 10
batch_size = 32
historico_loss = []

print("Iniciando treinamento do modelo LSTM...\n")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = torch.FloatTensor(y_train[i:i + batch_size])

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    media_loss = epoch_loss / n_batches
    historico_loss.append(media_loss)
    print(f"Época [{epoch + 1:02d}/{epochs}]  Loss: {media_loss:.6f}")

# ---------
#  Prever
# ---------
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# ---------------
#  Desnormalizar
# ---------------
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test)

# ----------------------
#  Métricas de avaliação
# ----------------------
rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
mae  = mean_absolute_error(y_test_real, predictions)
mape = np.mean(np.abs((y_test_real - predictions) / y_test_real)) * 100

print(f"\n--- Métricas do Modelo LSTM ---")
print(f"RMSE : R$ {rmse:.4f}")
print(f"MAE  : R$ {mae:.4f}")
print(f"MAPE : {mape:.2f}%")

# ----------------------------------------
#  Gráfico 1 — Previsão vs Valor Real
# ----------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_real,  label='Valor Real (PETR4)',         color='blue',   linewidth=1.5)
plt.plot(predictions,  label='Previsão do Modelo (LSTM)',  color='orange', linestyle='--', linewidth=1.5)
plt.title('Previsão de Fechamento da PETR4: Modelo LSTM vs Valor Real',
          fontsize=14, fontweight='bold')
plt.xlabel('Tempo (Dias do Conjunto de Teste)', fontsize=12)
plt.ylabel('Preço da Ação (R$)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('lstm_previsao_vs_real.png', dpi=150)
plt.show()

# ----------------------------------------
#  Gráfico 2 — Erro Absoluto ao longo do tempo
# ----------------------------------------
erro = np.abs(y_test_real - predictions)

plt.figure(figsize=(12, 4))
plt.plot(erro, label='Erro Absoluto (LSTM)', color='orange', linewidth=1)
plt.title('Erro Absoluto entre Previsão LSTM e Valor Real', fontsize=13, fontweight='bold')
plt.xlabel('Tempo (Dias do Conjunto de Teste)', fontsize=12)
plt.ylabel('Erro (R$)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('lstm_erro_absoluto.png', dpi=150)
plt.show()

# ----------------------------------------
#  Gráfico 3 — Curva de Loss do Treinamento
# ----------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), historico_loss, marker='o', color='darkorange', linewidth=2)
plt.title('Curva de Loss — Treinamento do Modelo LSTM', fontsize=13, fontweight='bold')
plt.xlabel('Época', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('lstm_curva_loss.png', dpi=150)
plt.show()