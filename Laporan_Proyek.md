# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek

Churn pelanggan merupakan tantangan besar di industri telekomunikasi. Perusahaan menghadapi kerugian ketika pelanggan berhenti menggunakan layanan mereka. Biaya mendapatkan pelanggan baru bisa lima kali lebih tinggi dibanding mempertahankan pelanggan lama (McKinsey & Company, 2020). Oleh karena itu, memprediksi churn dengan machine learning memiliki nilai strategis yang tinggi.

Dalam proyek ini, digunakan pendekatan klasifikasi untuk memprediksi kemungkinan seorang pelanggan berhenti berlangganan (churn) berdasarkan atribut layanan dan demografis.

📚 Referensi:

- McKinsey & Company. (2020). [The Value of Customer Retention](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-cs-of-customer-satisfaction-consistency-consistency-consistency)
- Ahmed, A., & Maheswaran, M. (2019). A Machine Learning Approach to Customer Churn Prediction in Telecom Industry.
- IBM. (n.d.). [Predicting Customer Churn with IBM Watson](https://www.ibm.com/blogs/watson-health/predicting-customer-churn/)

---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) atau tidak berdasarkan data historis pelanggan?
2. Algoritma machine learning apa yang memberikan hasil terbaik dalam memprediksi churn?

### Goals

1. Mengembangkan model machine learning untuk memprediksi pelanggan berisiko churn.
2. Membandingkan beberapa algoritma untuk memilih model terbaik.

### Solution Statements

- Menggunakan model klasifikasi: Logistic Regression, Random Forest, dan XGBoost.
- Evaluasi menggunakan metrik: akurasi, precision, recall, F1-score, dan ROC-AUC.

---

## Data Understanding

**Sumber Data:**
[Telco Customer Churn – IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/)

**Ukuran Dataset:**
- 7043 baris (pelanggan)
- 21 kolom (fitur)

**Kondisi Data:**
- Missing value: 11 nilai kosong pada kolom `TotalCharges`.
- Duplikat: Tidak ditemukan duplikat berdasarkan `customerID`.
- Outlier: Beberapa nilai tinggi pada `MonthlyCharges`, namun masih masuk akal secara bisnis.

**Deskripsi Fitur:**

| Fitur            | Deskripsi |
|------------------|----------|
| customerID       | ID unik pelanggan |
| gender           | Jenis kelamin pelanggan |
| SeniorCitizen    | 1 jika pelanggan lansia, 0 jika tidak |
| Partner          | Apakah memiliki pasangan |
| Dependents       | Apakah memiliki tanggungan |
| tenure           | Lama berlangganan (bulan) |
| PhoneService     | Apakah menggunakan layanan telepon |
| MultipleLines    | Apakah menggunakan lebih dari satu jalur telepon |
| InternetService  | Jenis layanan internet |
| OnlineSecurity   | Apakah menggunakan keamanan online |
| OnlineBackup     | Apakah menggunakan backup online |
| DeviceProtection | Proteksi perangkat |
| TechSupport      | Dukungan teknis |
| StreamingTV      | Layanan TV streaming |
| StreamingMovies  | Layanan film streaming |
| Contract         | Jenis kontrak (bulanan, tahunan, dll.) |
| PaperlessBilling | Apakah menggunakan tagihan elektronik |
| PaymentMethod    | Metode pembayaran |
| MonthlyCharges   | Biaya bulanan |
| TotalCharges     | Total biaya sepanjang waktu berlangganan |
| Churn            | Target (1 = churn, 0 = tidak churn) |

---

## Data Preparation

Langkah-langkah yang dilakukan:

1. **Menghapus Kolom Tidak Relevan:**
   - `customerID` dihapus karena tidak memberikan nilai prediktif.

2. **Menangani Missing Value:**
   - Nilai kosong pada `TotalCharges` diisi dengan median.

3. **Transformasi Kolom:**
   - `TotalCharges` dikonversi menjadi numerik karena awalnya berbentuk string.

4. **Encoding Fitur Kategorikal:**
   - Menggunakan One-Hot Encoding untuk fitur seperti `Contract`, `InternetService`, dll.

5. **Normalisasi Fitur Numerik:**
   - `MonthlyCharges`, `TotalCharges`, dan `tenure` dinormalisasi dengan MinMaxScaler.

6. **Splitting Data:**
   - Dataset dibagi menjadi data latih dan uji (80%:20%).

---

## Modeling

Tiga model digunakan:

### 1. Logistic Regression
- Model linear dasar untuk klasifikasi.
- Parameter: `penalty='l2'`, `solver='liblinear'`
- Kelebihan: cepat, interpretatif.

### 2. Random Forest Classifier
- Ensemble dari decision trees.
- Parameter: `n_estimators=100`, `max_depth=None`
- Kelebihan: akurasi tinggi, tahan terhadap overfitting.

### 3. XGBoost Classifier
- Model boosting yang fokus pada kesalahan model sebelumnya.
- Parameter: `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`
- Kelebihan: performa tinggi untuk dataset tabular.

---

## Evaluation

Metrik yang digunakan:

- **Accuracy**: rasio prediksi benar.
- **Precision**: ketepatan prediksi churn.
- **Recall**: seberapa banyak churn yang terdeteksi.
- **F1-Score**: rata-rata harmonis antara precision dan recall.
- **ROC-AUC**: area di bawah kurva ROC.

### Hasil Evaluasi:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80     | 0.65      | 0.55   | 0.59     | 0.82    |
| Random Forest       | 0.83     | 0.70      | 0.60   | 0.64     | 0.85    |
| XGBoost             | 0.85     | 0.72      | 0.65   | 0.68     | 0.88    |

### Kesimpulan:

Model terbaik adalah **XGBoost**, dengan skor terbaik pada semua metrik. Model ini bisa digunakan untuk memprioritaskan pelanggan berisiko churn untuk intervensi dini.

---

## Inference

Contoh penggunaan model untuk memprediksi pelanggan:

```python
# Load model
loaded_model = joblib.load(model_path)

# Prediksi pada 5 sampel
sample = X_test.sample(5, random_state=1)
predictions = loaded_model.predict(sample)
probs = loaded_model.predict_proba(sample)[:, 1]

# Hasil
result_df = pd.DataFrame({
    'Predicted': predictions,
    'Probability (Churn)': probs,
    'Actual': y_test.loc[sample.index].values
}, index=sample.index)

print(result_df)
