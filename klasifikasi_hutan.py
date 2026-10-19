
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)

# 1. Membaca dataset
df = pd.read_csv("data_hutan.csv")

print("=" * 55)
print("KLASIFIKASI GANGGUAN TUTUPAN HUTAN")
print("=" * 55)

print("\nLima data pertama:")
print(df.head())

print("\nJumlah data:", len(df))
print("\nJumlah data kosong:")
print(df.isnull().sum())

# 2. Menyiapkan fitur dan label
fitur = [
    "perubahan_vegetasi",
    "perubahan_kelembapan",
    "akses_jalan",
    "aktivitas_pembukaan"
]

X = df[fitur]
y = df["kategori"]

# 3. Membagi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Melatih model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 5. Menguji model
y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nAkurasi model:", round(akurasi, 3))
print("Macro F1-score:", round(macro_f1, 3))

print("\nLaporan klasifikasi:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 6. Membuat confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=model.classes_,
    xticks_rotation=45,
    cmap="Greens",
    ax=ax
)

ax.set_title("Evaluasi Klasifikasi Gangguan Hutan")
fig.tight_layout()
fig.savefig("hasil_evaluasi.png", dpi=150)
plt.close(fig)

print("\nGrafik disimpan sebagai hasil_evaluasi.png")

# 7. Melihat fitur yang berpengaruh pada model
kepentingan = pd.DataFrame({
    "Fitur": fitur,
    "Nilai Kepentingan": model.feature_importances_
}).sort_values(
    by="Nilai Kepentingan",
    ascending=False
)

print("\nKepentingan fitur menurut model:")
print(kepentingan.to_string(index=False))

# 8. Prediksi data simulasi baru
data_baru = pd.DataFrame([{
    "perubahan_vegetasi": 0.82,
    "perubahan_kelembapan": 0.32,
    "akses_jalan": 0.78,
    "aktivitas_pembukaan": 0.86
}])

prediksi = model.predict(data_baru)[0]
print("\nPrediksi untuk data simulasi baru:", prediksi)

print("\nCatatan: seluruh data dan prediksi merupakan simulasi.")
