
# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek

Churn pelanggan merupakan tantangan besar dalam industri telekomunikasi. Ketika pelanggan berhenti berlangganan, perusahaan menghadapi kehilangan pendapatan dan harus mengeluarkan biaya tambahan untuk memperoleh pelanggan baru. Oleh karena itu, mampu memprediksi pelanggan yang berisiko churn dapat memberikan keuntungan strategis.

Menurut studi McKinsey & Company (2020), biaya untuk menarik pelanggan baru bisa 5 kali lebih besar dibanding mempertahankan pelanggan lama. Dengan memanfaatkan data historis dan pendekatan machine learning, perusahaan dapat mengidentifikasi pelanggan yang berpotensi churn dan memberikan intervensi yang tepat.

📚 Referensi:
- McKinsey & Company. (2020). *The Value of Customer Retention*. https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-cs-of-customer-satisfaction-consistency-consistency-consistency  
- Ahmed, A., & Maheswaran, M. (2019). *A Machine Learning Approach to Customer Churn Prediction in Telecom Industry*. International Journal of Computer Applications, 178(7), 1–6.  
- IBM. (n.d.). *Predicting Customer Churn with IBM Watson*. https://www.ibm.com/blogs/watson-health/predicting-customer-churn/

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) berdasarkan atribut perilaku dan layanan yang digunakan?
- Algoritma machine learning apa yang memberikan hasil terbaik dalam memprediksi churn pelanggan?

### Goals
- Mengembangkan model machine learning untuk mengklasifikasikan pelanggan yang akan churn atau tidak.
- Membandingkan beberapa algoritma untuk menentukan model terbaik berdasarkan metrik evaluasi yang relevan.

### Solution Statements
- Menerapkan lima algoritma klasifikasi: Logistic Regression, Random Forest, Support Vector Machine, XGBoost, dan K-Nearest Neighbors.
- Mengevaluasi performa tiap model menggunakan metrik: Accuracy, Precision, Recall, F1-score, dan ROC-AUC.
- Menentukan model terbaik untuk digunakan dalam proses prediksi churn operasional.

## Data Understanding

Dataset yang digunakan adalah [Telco Customer Churn dari IBM via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Dataset ini berisi informasi demografis dan penggunaan layanan pelanggan, serta status apakah pelanggan tersebut churn.

### Struktur Data:
- Jumlah baris: 7043
- Jumlah kolom: 21

### Kondisi Data:
- Terdapat nilai kosong pada kolom `TotalCharges` sebanyak 11 baris.
- Tidak ditemukan data duplikat.
- Outlier tidak dihapus karena masih dalam rentang yang wajar secara domain.

### Fitur-Fitur:
- `customerID`: ID unik pelanggan.
- `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`: fitur kategorikal.
- `SeniorCitizen`: biner, 1 untuk lansia, 0 bukan.
- `tenure`: lamanya pelanggan berlangganan (bulan).
- `MonthlyCharges`, `TotalCharges`: jumlah biaya yang dibayarkan.
- `Churn`: target label.

## Data Preparation

Tahapan yang dilakukan untuk persiapan data adalah:

1. **Menghapus Missing Values**: Kolom `TotalCharges` dikonversi ke numerik dan nilai NaN dihapus (11 baris).
2. **Drop Kolom Tidak Relevan**: Kolom `customerID` dihapus karena tidak memiliki kontribusi prediktif.
3. **Encoding Label**: Label `Churn` diubah menjadi 1 (Yes) dan 0 (No).
4. **One-Hot Encoding**: Seluruh fitur kategorikal diubah menggunakan teknik one-hot encoding.
5. **Normalisasi**: Fitur numerik (`tenure`, `MonthlyCharges`, `TotalCharges`) dinormalisasi menggunakan `MinMaxScaler`.
6. **Split Data**: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan stratifikasi label.

## Modeling

Lima model machine learning digunakan:

1. **Logistic Regression**
   - Model linier dasar untuk klasifikasi.
   - Parameter: `max_iter=1000`
   - Kelebihan: cepat, mudah diinterpretasi.
   - Kekurangan: kurang fleksibel untuk data kompleks.

2. **Random Forest**
   - Ansambel dari pohon keputusan.
   - Parameter: `n_estimators=100`, `random_state=42`
   - Kelebihan: kuat terhadap overfitting, performa stabil.
   - Kekurangan: interpretasi sulit.

3. **Support Vector Machine (SVM)**
   - Memisahkan kelas menggunakan hyperplane.
   - Parameter: kernel default (RBF)
   - Kelebihan: akurat pada ruang fitur tinggi.
   - Kekurangan: lambat di dataset besar.

4. **XGBoost**
   - Gradient boosting tree.
   - Parameter: `use_label_encoder=False`, `eval_metric='logloss'`
   - Kelebihan: performa tinggi, menangani missing value.
   - Kekurangan: interpretasi lebih sulit.

5. **K-Nearest Neighbors (KNN)**
   - Klasifikasi berdasarkan tetangga terdekat.
   - Parameter: `n_neighbors=5`
   - Kelebihan: intuitif, non-parametrik.
   - Kekurangan: sensitif terhadap skala dan noise.

Setiap model dilatih dan disimpan untuk evaluasi lanjutan.

## Evaluation

Model dievaluasi menggunakan metrik berikut:

- **Accuracy**: Proporsi prediksi benar.
- **Precision**: Proporsi churn yang diprediksi benar.
- **Recall**: Kemampuan menangkap seluruh churn aktual.
- **F1-score**: Harmonis antara precision dan recall.
- **ROC-AUC**: Luas area di bawah kurva ROC.

### Hasil Evaluasi:

| Model                 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | **0.80** | **0.65**  | **0.57** | **0.61** | **0.83** |
| Random Forest         | 0.79     | 0.62      | 0.51   | 0.56     | 0.82     |
| SVM                   | 0.79     | 0.63      | 0.52   | 0.57     | 0.79     |
| XGBoost               | 0.77     | 0.57      | 0.52   | 0.54     | 0.81     |
| K-Nearest Neighbors   | 0.75     | 0.53      | 0.57   | 0.55     | 0.77     |

### Kesimpulan:
Logistic Regression merupakan model terbaik berdasarkan keseimbangan antara metrik. Selain itu, model ini juga lebih mudah diinterpretasikan oleh tim bisnis dan teknis.

## Inference

Setelah pelatihan dan evaluasi, model Logistic Regression disimpan dan digunakan untuk melakukan prediksi (inference) terhadap data pelanggan baru.

### Contoh Kode Inference:

```python
# Load model yang telah disimpan
loaded_model = joblib.load(model_path)

# Ambil 5 sampel dari data uji
sample = X_test.sample(5, random_state=1)
true_labels = y_test.loc[sample.index]

# Prediksi kelas dan probabilitas
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

### Contoh Hasil:

| Index | Predicted | Probability (Churn) | Actual |
|-------|-----------|---------------------|--------|
| 458   | 0         | 0.18                | 0      |
| 3327  | 0         | 0.14                | 0      |
| 5104  | 0         | 0.03                | 0      |
| 5089  | 0         | 0.43                | 0      |
| 3377  | 0         | 0.005               | 0      |

## Dampak terhadap Bisnis

- **Problem statement terjawab**: Ya, model berhasil memprediksi churn.
- **Goal tercapai**: Ya, model terbaik berhasil diidentifikasi.
- **Solusi berdampak**: Model dapat digunakan oleh perusahaan untuk menargetkan pelanggan berisiko dengan strategi retensi khusus, seperti diskon atau peningkatan layanan.

