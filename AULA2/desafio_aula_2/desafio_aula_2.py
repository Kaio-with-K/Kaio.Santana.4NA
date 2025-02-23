# Implementação de regressão linear simples utilizando o método dos mínimos quadrados

# 1. Definir as listas de valores de x e y (mesmo exemplo passado)
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# 2. Calcular as médias de x e y (y= β0 + β1x)
# 𝛽 1 = ∑ 𝑖 = 1 𝑛 ( 𝑥 𝑖 − 𝑥 ˉ ) ( 𝑦 𝑖 − 𝑦 ˉ ) ∑ 𝑖 = 1 𝑛 ( 𝑥 𝑖 − 𝑥 ˉ ) 2 β 1 ​ = ∑ i=1 n ​ (x i ​ − x ˉ ) 2 ∑ i=1 n ​ (x i ​ − x ˉ )(y i ​ − y ˉ ​ ) ​
# 𝛽 0 = 𝑦 ˉ − 𝛽 1 𝑥 ˉ β 0 ​ = y ˉ ​ −β 1 ​ x ˉ 
media_x = sum(x) / len(x)
media_y = sum(y) / len(y)

# 3. Inicializar as variáveis para os somatórios:
#    - soma_numerador: soma de (x_i - média_x) * (y_i - média_y)
#    - soma_denominador: soma de (x_i - média_x)²
soma_numerador = 0
soma_denominador = 0

# 4. Calcular os somatórios utilizando um loop
for i in range(len(x)):
    soma_numerador += (x[i] - media_x) * (y[i] - media_y)
    soma_denominador += (x[i] - media_x) ** 2

# 5. Calcular os coeficientes beta1 (inclinação) e beta0 (intercepto)
beta1 = soma_numerador / soma_denominador
beta0 = media_y - beta1 * media_x

# 6. Imprimir os resultados da reta de regressão (y = beta0 + beta1 * x)
print("Coeficiente beta0 (intercepto):", beta0)
print("Coeficiente beta1 (inclinação):", beta1)
