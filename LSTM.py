import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch

import random

from rnn import LSTMModel, create_dataset, treinar, prever, calcular_metricas

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#  Dados da Petrobrás de 2016 a 2026

df = yf.download("PETR4.SA", start="2016-01-01", end="2026-01-01")
df = df[['Close']]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

X, y = create_dataset(df_scaled)

split = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split])
X_test  = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split])
y_test  = y[split:]   

model = LSTMModel(hidden_size=50, num_layers=2, dropout=0.2)

epochs     = 30
batch_size = 32

print("Treinamento do modelo LSTM\n")
historico_loss = treinar(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

predictions, y_test_real = prever(model, X_test, y_test, scaler)

print(f"\nMétricas do Modelo LSTM")
rmse, mae, mape = calcular_metricas(y_test_real, predictions, "LSTM")


#  Gráfico 1 — Previsão vs Valor Real

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


#  Gráfico 2 — Erro Absoluto ao longo do tempo

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


#  Gráfico 3 — Curva de Loss do Treinamento

plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), historico_loss, marker='o', color='darkorange', linewidth=2)
plt.title('Curva de Loss — Treinamento do Modelo LSTM', fontsize=13, fontweight='bold')
plt.xlabel('Época', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('lstm_curva_loss.png', dpi=150)
plt.show()