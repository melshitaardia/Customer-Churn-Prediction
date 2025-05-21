
# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek

Proyek ini berada dalam domain Telekomunikasi. Industri telekomunikasi memiliki tantangan besar dalam mempertahankan pelanggan karena tingginya tingkat perpindahan pelanggan (customer churn). Mendeteksi churn lebih awal memungkinkan perusahaan melakukan intervensi dan meningkatkan loyalitas pelanggan.

Churn dapat mengakibatkan kerugian besar bagi perusahaan. Menurut riset dari Frederick Reichheld (Bain & Company), meningkatkan retensi pelanggan sebesar 5% dapat meningkatkan keuntungan hingga 25–95% (Reichheld, 2001). Oleh karena itu, deteksi churn secara proaktif melalui machine learning dapat menjadi solusi efektif.

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) atau tidak berdasarkan data historis pelanggan?
2. Algoritma machine learning apa yang memberikan hasil terbaik dalam memprediksi churn?

### Goals

1. Mengembangkan model machine learning yang mampu memprediksi pelanggan yang berpotensi churn.
2. Membandingkan beberapa algoritma untuk menentukan model terbaik.

### Solution Statements

- Menggunakan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan XGBoost.
- Melakukan evaluasi berdasarkan metrik seperti akurasi, precision, recall, F1-score, dan AUC untuk memilih model terbaik.

## Data Understanding

Dataset yang digunakan adalah Telco Customer Churn dari IBM Sample Data Sets. Dataset ini berisi data pelanggan termasuk layanan yang digunakan, informasi pembayaran, dan apakah pelanggan berhenti (churn).

Beberapa fitur dalam dataset:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`: informasi demografis.
- `tenure`, `PhoneService`, `InternetService`, `Contract`: informasi layanan.
- `MonthlyCharges`, `TotalCharges`: informasi pembayaran.
- `Churn`: target label, 1 jika churn dan 0 jika tidak.

Sumber: IBM Sample Data Sets – Telco Customer Churn

## Data Preparation

- Menghapus kolom tidak relevan seperti `customerID`.
- Mengubah kolom `TotalCharges` ke numerik dan menangani missing value.
- Melakukan one-hot encoding untuk fitur kategorikal.
- Normalisasi fitur numerik dengan MinMaxScaler.
- Memisahkan data menjadi fitur (X) dan target (y).
- Split data menjadi data latih dan data uji (80:20).

## Modeling

Model yang digunakan:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Setiap model dilatih menggunakan data training. Hasil evaluasi digunakan untuk memilih model terbaik.

## Evaluation

Metrik yang digunakan:

- Akurasi: proporsi prediksi benar terhadap seluruh data.
- Precision: ketepatan prediksi positif (churn).
- Recall: kemampuan mendeteksi semua churn.
- F1-score: harmonisasi antara precision dan recall.
- ROC-AUC: luas di bawah kurva ROC, menggambarkan kemampuan model membedakan churn dan tidak.

Visualisasi ROC dan Confusion Matrix disediakan untuk memperjelas performa masing-masing model.

Hasil menunjukkan bahwa XGBoost memberikan hasil terbaik dengan keseimbangan antara precision dan recall.

## Inference

```python
# Load model yang telah disimpan
loaded_model = joblib.load(model_path)

# Contoh inference: Ambil 5 sample dari test set
sample = X_test.sample(5, random_state=1)
true_labels = y_test.loc[sample.index]

# Prediksi
predictions = loaded_model.predict(sample)
probs = loaded_model.predict_proba(sample)[:, 1]

# Tampilkan hasil
result_df = pd.DataFrame({
    'Predicted': predictions,
    'Probability (Churn)': probs,
    'Actual': true_labels.values
}, index=sample.index)

print("Hasil Inference:")
print(result_df)
```

Contoh hasil inference:

| Index | Predicted | Probability (Churn) | Actual |
|-------|-----------|---------------------|--------|
| 458   |     0     |         0.28        |   0    |
| 3327  |     0     |         0.38        |   0    |
| 5104  |     0     |         0.00        |   0    |
| 5089  |     0     |         0.33        |   0    |
| 3377  |     0     |         0.01        |   0    |

Interpretasi:

Model memprediksi semua pelanggan ini tidak akan churn (label 0), dan semua prediksi cocok dengan label aktual. Nilai probabilitas churn menunjukkan keyakinan model, dan bisa digunakan untuk menentukan threshold intervensi. Misalnya, jika probabilitas di atas 0.4, perusahaan dapat mulai mempertimbangkan strategi retensi meskipun pelanggan belum pasti churn.

---
