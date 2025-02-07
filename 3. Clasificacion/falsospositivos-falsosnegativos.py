# evaluar falsos positivos y negativos
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
tp, fp = cm[0, 0], cm[0, 1]
fn, tn = cm[1, 0], cm[1, 1]

# Cálculos
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Precisión (Accuracy): {accuracy:.2%}")
print(f"Tasa de Error: {error_rate:.2%}")
print(f"Tasa de Falsos Positivos (FPR): {fpr:.2%}")
print(f"Tasa de Falsos Negativos (FNR): {fnr:.2%}")
