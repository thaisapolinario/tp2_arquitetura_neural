import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import time

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
y_test  = y[split:]

# ----------------------
#  Definição dos Modelos
# ----------------------
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

# -----------------------------------------------
#  Função genérica de treino (usada pelos dois)
# -----------------------------------------------
def treinar(model, X_train, y_train, epochs=10, batch_size=32, lr=0.001):
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
            n_batches  += 1

        media = epoch_loss / n_batches
        historico_loss.append(media)
        print(f"  Época [{epoch + 1:02d}/{epochs}]  Loss: {media:.6f}")

    return historico_loss

# -----------------------------------------------
#  Treinar LSTM
# -----------------------------------------------
print("=" * 45)
print("  Treinando LSTM...")
print("=" * 45)
lstm_model = LSTMModel()
t0 = time.time()
lstm_loss = treinar(lstm_model, X_train, y_train)
lstm_tempo = time.time() - t0
print(f"  Tempo de treinamento LSTM: {lstm_tempo:.1f}s\n")

# -----------------------------------------------
#  Treinar GRU
# -----------------------------------------------
print("=" * 45)
print("  Treinando GRU...")
print("=" * 45)
gru_model = GRUModel()
t0 = time.time()
gru_loss = treinar(gru_model, X_train, y_train)
gru_tempo = time.time() - t0
print(f"  Tempo de treinamento GRU: {gru_tempo:.1f}s\n")

# -----------------------------------------------
#  Previsões
# -----------------------------------------------
def prever(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
    pred_real    = scaler.inverse_transform(pred)
    y_test_real  = scaler.inverse_transform(y_test)
    return pred_real, y_test_real

lstm_pred, y_real = prever(lstm_model, X_test, y_test, scaler)
gru_pred,  _      = prever(gru_model,  X_test, y_test, scaler)

# -----------------------------------------------
#  Métricas
# -----------------------------------------------
def calcular_metricas(y_real, y_pred, nome):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    print(f"--- Métricas {nome} ---")
    print(f"  RMSE : R$ {rmse:.4f}")
    print(f"  MAE  : R$ {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return rmse, mae, mape

print()
lstm_rmse, lstm_mae, lstm_mape = calcular_metricas(y_real, lstm_pred, "LSTM")
print()
gru_rmse,  gru_mae,  gru_mape  = calcular_metricas(y_real, gru_pred,  "GRU")

# -----------------------------------------------
#  Contagem de parâmetros
# -----------------------------------------------
def contar_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

lstm_params = contar_params(lstm_model)
gru_params  = contar_params(gru_model)

print(f"\n  Parâmetros LSTM : {lstm_params:,}")
print(f"  Parâmetros GRU  : {gru_params:,}")

# ================================================
#  GRÁFICOS DE COMPARAÇÃO
# ================================================

# -----------------------------------------------
#  Gráfico 1 — Previsão vs Real (ambos juntos)
# -----------------------------------------------
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

# -----------------------------------------------
#  Gráfico 2 — Curvas de Loss lado a lado
# -----------------------------------------------
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

# -----------------------------------------------
#  Gráfico 3 — Erro Absoluto lado a lado
# -----------------------------------------------
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

# -----------------------------------------------
#  Gráfico 4 — Tabela de métricas visual
# -----------------------------------------------
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

# Destacar cabeçalho
for j in range(3):
    tabela[0, j].set_facecolor('#2c3e50')
    tabela[0, j].set_text_props(color='white', fontweight='bold')

# Destacar melhor valor em cada linha (menor = melhor)
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

print("\nTodos os gráficos foram salvos como .png no diretório atual.")