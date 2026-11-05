import numpy as np
from numpy.random import randn

# N é o tamanho do batch; D_in é a dimensão de entrada;
# H é a dimensão oculta; D_out é a dimensão de saída.
N, D_in, H, D_out = 64, 1000, 100, 10

# Cria dados de entrada (x) e saída desejada (y) aleatórios
x, y = randn(N, D_in), randn(N, D_out)

# Inicializa os pesos da rede neural aleatoriamente
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
  # 1. Forward Pass: computa a predição y_pred
  # Utiliza a função de ativação sigmóide na camada oculta
  h = 1 / (1 + np.exp(-x.dot(w1)))
  y_pred = h.dot(w2)

  # 2. Calcula e exibe a função de perda (Loss - Erro Quadrático)
  loss = np.square(y_pred - y).sum()
  print(f"Época {t}, Loss: {loss}")

  # 3. Backward Pass: calcula os gradientes analíticos
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h.T.dot(grad_y_pred)
  grad_h = grad_y_pred.dot(w2.T)
  grad_w1 = x.T.dot(grad_h * h * (1 - h))

  # 4. Atualização dos pesos (Gradient Descent)
  w1 -= 1e-4 * grad_w1
  w2 -= 1e-4 * grad_w2
