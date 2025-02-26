#Realizando todas as importações necessárias
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Carregando o dataset Wine
data = load_wine()
X = data.data  # Features (Atributos do Vinho)
y = data.target  # Classe alvo (3 tipos de vinho)

# Dividindo os dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando os dados (KNN é sensível à escala das features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Testando diferentes valores de k para encontrar o melhor desempenho
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())

# Melhor valor de k
best_k = k_values[np.argmax(accuracy_scores)]
print(f"Melhor valor de k: {best_k}")

# Treinando o modelo com o melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Exibindo as métricas
print("\nMétricas do Modelo KNN:")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Matriz de Confusão:")
print(conf_matrix)

# Salvando a Matriz de Confusão 
np.savetxt("matriz_confusao.txt", conf_matrix, fmt='%d')

# Cálculo da Curva ROC e AUC (usando One-vs-Rest para multiclasse)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = knn.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

roc_data = []
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data.append((fpr, tpr, roc_auc))
    print(f"AUC para Classe {data.target_names[i]}: {roc_auc:.4f}")

# Salvando os dados da curva ROC 
with open("curva_roc.txt", "w") as f:
    for i, (fpr, tpr, roc_auc) in enumerate(roc_data):
        f.write(f"Classe {data.target_names[i]}\n")
        f.write(f"FPR: {list(fpr)}\n")
        f.write(f"TPR: {list(tpr)}\n")
        f.write(f"AUC: {roc_auc:.4f}\n\n")
