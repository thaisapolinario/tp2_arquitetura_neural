import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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



# -------------------------
#  Normalizar dados (0 a 1)
# --------------------------

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)



# ----------------
#  Criar janelas
# ----------------

def create_dataset(data, time_step = 60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : i+time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(df_scaled)



# ------------------------
#  Separar treino e teste
# ------------------------

split = int(len(X) * 0.8)

# Convertendo para Tensores do PyTorch
X_train = torch.FloatTensor(X[:split])
X_test = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split])
y_test = y[split:] # Mantido em numpy para o cálculo final



# ------------------------------
#  Criar modelo LSTM em PyTorch
# ------------------------------

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.linear = nn.Linear(50, 1)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.linear(out[:, -1, :]) # Pega apenas o último passo temporal
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# ----------
#  Treinar
# ----------

epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = torch.FloatTensor(y_train[i:i+batch_size])
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()



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



# -------------------------------
#  Plotar Previsão vs Valor Real
# -------------------------------

rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
print(f"RMSE: {rmse}")

# Criar o gráfico comparativo
plt.figure(figsize=(12, 6))

# Linha 1: Valores Reais
plt.plot(y_test_real, label='Valor Real (PETR4)', color='blue', linewidth=1.5)

# Linha 2: Valores Previstos pelo Modelo
plt.plot(predictions, label='Previsão do Modelo (LSTM)', color='orange', linestyle='--', linewidth=1.5)

# Customizações do gráfico
plt.title('Previsão de Fechamento da PETR4: Modelo LSTM vs Valor Real', fontsize=14, fontweight='bold')
plt.xlabel('Tempo (Dias do Conjunto de Teste)', fontsize=12)
plt.ylabel('Preço da Ação (R$)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Exibir o gráfico na tela
plt.show()



# ---------------
#  Calcular erro
# ---------------

rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
print(f"RMSE: {rmse}")

# Calcular erro absoluto
erro = np.abs(y_test_real - predictions)

plt.figure()
plt.plot(erro, label='Erro absoluto')
plt.title('Erro entre previsão e valor real')
plt.xlabel('Tempo')
plt.ylabel('Erro (R$)')
plt.legend()
plt.show()