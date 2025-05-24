# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek

Churn pelanggan adalah masalah yang sangat penting dalam industri telekomunikasi, di mana perusahaan sering kehilangan pelanggan secara tiba-tiba dan tanpa peringatan. Dengan memahami pola-pola yang menyebabkan pelanggan berhenti menggunakan layanan, perusahaan dapat mengambil langkah-langkah proaktif untuk mempertahankan pelanggan dan meningkatkan keuntungan.

Menurut studi oleh McKinsey & Company (2020), biaya untuk mendapatkan pelanggan baru bisa lima kali lebih tinggi daripada mempertahankan pelanggan yang ada. Oleh karena itu, memprediksi churn menggunakan machine learning bisa memberikan manfaat strategis dalam pengambilan keputusan bisnis.

Dalam proyek ini, digunakan pendekatan klasifikasi untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (churn) berdasarkan atribut perilaku dan layanan yang digunakan.

📚 Referensi:

- McKinsey & Company. (2020). The Value of Customer Retention. https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-cs-of-customer-satisfaction-consistency-consistency-consistency  
- Ahmed, A., & Maheswaran, M. (2019). A Machine Learning Approach to Customer Churn Prediction in Telecom Industry. International Journal of Computer Applications, 178(7), 1–6.  
- IBM. (n.d.). Predicting Customer Churn with IBM Watson. https://www.ibm.com/blogs/watson-health/predicting-customer-churn/

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

Dataset yang digunakan adalah **Telco Customer Churn** dari IBM Sample Data Sets, tersedia di Kaggle:  
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset ini berisi data pelanggan termasuk informasi demografis, layanan yang digunakan, dan status churn.

### Jumlah Data

- Jumlah baris: 7.043
- Jumlah kolom: 21

### Kondisi Data

- Missing value ditemukan di kolom `TotalCharges` (11 nilai kosong), ditangani dengan menghapus baris terkait.
- Tidak terdapat data duplikat.
- Outlier tidak dihapus karena dianggap masih wajar secara domain.

### Uraian Fitur

- `customerID`: ID unik pelanggan.
- `gender`: jenis kelamin pelanggan.
- `SeniorCitizen`: apakah pelanggan adalah warga lanjut usia (0 = tidak, 1 = ya).
- `Partner`, `Dependents`: status hubungan dan tanggungan.
- `tenure`: lama berlangganan (dalam bulan).
- `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: jenis layanan yang digunakan.
- `Contract`, `PaperlessBilling`, `PaymentMethod`: informasi kontrak dan metode pembayaran.
- `MonthlyCharges`, `TotalCharges`: biaya bulanan dan total biaya.
- `Churn`: target prediksi (Yes/No).

## Data Preparation

Langkah-langkah persiapan data dilakukan sebagai berikut:

1. **Menghapus Kolom Tidak Relevan**
   - Kolom `customerID` dihapus karena tidak memiliki informasi prediktif.

2. **Menangani Missing Value**
   - Nilai kosong pada kolom `TotalCharges` dihapus (jumlah: 11 baris).

3. **Transformasi Tipe Data**
   - Kolom `TotalCharges` dikonversi dari `object` ke `float`.

4. **Encoding Fitur Kategorikal**
   - Label encoding digunakan untuk `SeniorCitizen`, `Churn`.
   - One-hot encoding digunakan untuk kolom kategorikal lain seperti `Contract`, `PaymentMethod`, dll.

5. **Normalisasi**
   - Fitur numerik `tenure`, `MonthlyCharges`, dan `TotalCharges` dinormalisasi menggunakan `MinMaxScaler`.

6. **Split Data**
   - Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan stratifikasi berdasarkan label `Churn`.

## Modeling

### 1. Logistic Regression
- Model klasifikasi linier sederhana untuk baseline.
- Parameter utama: default (`solver='lbfgs'`), `max_iter=1000`.

### 2. Random Forest Classifier
- Model ansambel berbasis decision tree.
- Parameter: `n_estimators=100`, `random_state=42`.

### 3. Support Vector Machine (SVM)
- Cocok untuk klasifikasi dengan margin maksimum.
- Kernel: RBF (default).
  
### 4. XGBoost Classifier
- Gradient boosting yang kuat dan efisien.
- Parameter: `use_label_encoder=False`, `eval_metric='logloss'`.

### 5. K-Nearest Neighbors (KNN)
- Model berbasis kedekatan jarak antar data.
- Parameter: `n_neighbors=5`.

Setiap model dilatih menggunakan data latih, dan dievaluasi di data uji.

## Evaluation

Metrik evaluasi:

- **Accuracy**: proporsi prediksi benar.
- **Precision**: proporsi prediksi churn yang benar.
- **Recall**: proporsi churn yang berhasil terdeteksi.
- **F1-score**: harmonisasi antara precision dan recall.
- **ROC-AUC**: seberapa baik model membedakan churn dan tidak churn.

| Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------- |----------|-----------|--------|----------|---------|
| **Logistic Regression**|     **0.80** |      **0.65** |   **0.57** |     **0.61** |    **0.83** |
| Random Forest      |     0.79 |      0.62 |   0.51 |     0.56 |    0.82 |
| SVM                |     0.79 |      0.63 |   0.52 |     0.57 |    0.79 |
| XGBoost            |     0.77 |      0.57 |   0.52 |     0.54 |    0.81 |
| KNN                |     0.75 |      0.53 |   0.57 |     0.55 |    0.77 |

Logistic Regression dipilih sebagai model terbaik karena menghasilkan metrik paling seimbang.

### Dampak Terhadap Business Understanding

- **Problem Statement Terjawab**: Ya, model mampu memprediksi churn berdasarkan fitur historis.
- **Goal Tercapai**: Ya, XGBoost terbukti menjadi model terbaik dengan performa solid.
- **Solusi Berdampak**: Ya, model ini dapat digunakan untuk mengidentifikasi pelanggan berisiko tinggi dan merancang strategi retensi seperti penawaran khusus.

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

### Contoh hasil:
Hasil Inference:
|                   | Pedicted | Probability (Churn) | Actual | 
|-------------------|----------|---------------------|--------|
|               458 |        0 |                0.18 |      0 | 
|              3327 |        0 |                0.14 |      0 |
|              5104 |        0 |                0.03 |      0 |
|              5089 |        0 |                0.43 |      0 |
|              3377 |        0 |               0.005 |      0 |  
