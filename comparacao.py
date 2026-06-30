import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import time

from rnn import (
    LSTMModel, GRUModel, create_dataset,
    treinar, prever, calcular_metricas, contar_params
)


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


#  Treinar LSTM

print("=" * 45)
print("  Treinando LSTM...")
print("=" * 45)
lstm_model = LSTMModel()
t0 = time.time()
lstm_loss = treinar(lstm_model, X_train, y_train)
lstm_tempo = time.time() - t0
print(f"  Tempo de treinamento LSTM: {lstm_tempo:.1f}s\n")


#  Treinar GRU

print("=" * 45)
print("  Treinando GRU...")
print("=" * 45)
gru_model = GRUModel()
t0 = time.time()
gru_loss = treinar(gru_model, X_train, y_train)
gru_tempo = time.time() - t0
print(f"  Tempo de treinamento GRU: {gru_tempo:.1f}s\n")


#  Previsões

lstm_pred, y_real = prever(lstm_model, X_test, y_test, scaler)
gru_pred,  _      = prever(gru_model,  X_test, y_test, scaler)


#  Métricas

print()
lstm_rmse, lstm_mae, lstm_mape = calcular_metricas(y_real, lstm_pred, "LSTM")
print()
gru_rmse,  gru_mae,  gru_mape  = calcular_metricas(y_real, gru_pred,  "GRU")


lstm_params = contar_params(lstm_model)
gru_params  = contar_params(gru_model)

print(f"\n  Parâmetros LSTM : {lstm_params:,}")
print(f"  Parâmetros GRU  : {gru_params:,}")



#  Gráfico 1 — Previsão vs Real 

plt.figure(figsize=(14, 6))
plt.plot(y_real,    label='Valor Real (PETR4)', color='blue',   linewidth=1.5)
plt.plot(lstm_pred, label='LSTM',               color='orange', linestyle='--', linewidth=1.5)
plt.plot(gru_pred,  label='GRU',                color='red',    linestyle=':',  linewidth=1.5)
plt.title('PETR4 — Previsão LSTM vs GRU vs Valor Real', fontsize=14, fontweight='bold')
plt.xlabel('Tempo (Dias do Conjunto de Teste)', fontsize=12)
plt.ylabel('Preço da Ação (R$)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('comparacao_previsao.png', dpi=150)
plt.show()


#  Gráfico 2 — Curvas de Loss lado a lado

epochs = range(1, len(lstm_loss) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

axes[0].plot(epochs, lstm_loss, marker='o', color='darkorange', linewidth=2)
axes[0].set_title('Curva de Loss — LSTM', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Época', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].plot(epochs, gru_loss, marker='o', color='darkred', linewidth=2)
axes[1].set_title('Curva de Loss — GRU', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Época', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Comparação das Curvas de Loss', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacao_loss.png', dpi=150)
plt.show()

#  Gráfico 3 — Erro Absoluto lado a lado

lstm_erro = np.abs(y_real - lstm_pred)
gru_erro  = np.abs(y_real - gru_pred)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

axes[0].plot(lstm_erro, color='orange', linewidth=1)
axes[0].set_title('Erro Absoluto — LSTM', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Erro (R$)', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].plot(gru_erro, color='red', linewidth=1)
axes[1].set_title('Erro Absoluto — GRU', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tempo (Dias do Conjunto de Teste)', fontsize=11)
axes[1].set_ylabel('Erro (R$)', fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Comparação do Erro Absoluto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacao_erro.png', dpi=150)
plt.show()


#  Gráfico 4 — Tabela de métricas visual

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')

tabela_dados = [
    ['RMSE (R$)',      f'{lstm_rmse:.4f}',  f'{gru_rmse:.4f}'],
    ['MAE (R$)',       f'{lstm_mae:.4f}',   f'{gru_mae:.4f}'],
    ['MAPE (%)',       f'{lstm_mape:.2f}',  f'{gru_mape:.2f}'],
    ['Parâmetros',    f'{lstm_params:,}',  f'{gru_params:,}'],
    ['Tempo treino',  f'{lstm_tempo:.1f}s', f'{gru_tempo:.1f}s'],
]

tabela = ax.table(
    cellText=tabela_dados,
    colLabels=['Métrica', 'LSTM', 'GRU'],
    cellLoc='center',
    loc='center'
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.4, 2.0)

for j in range(3):
    tabela[0, j].set_facecolor('#2c3e50')
    tabela[0, j].set_text_props(color='white', fontweight='bold')


for i, row in enumerate(tabela_dados):
    try:
        val_lstm = float(row[1].replace(',', '').replace('s', ''))
        val_gru  = float(row[2].replace(',', '').replace('s', ''))
        melhor = 1 if val_lstm <= val_gru else 2
        tabela[i + 1, melhor].set_facecolor('#d5f5e3')  # Verde claro
    except ValueError:
        pass

plt.title('Comparação de Métricas: LSTM vs GRU', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('comparacao_tabela.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTodos os gráficos foram salvos")