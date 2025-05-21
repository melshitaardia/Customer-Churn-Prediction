# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek

Customer churn atau kehilangan pelanggan adalah masalah umum dalam industri telekomunikasi. Kemampuan untuk memprediksi pelanggan mana yang kemungkinan akan berhenti berlangganan sangat penting agar perusahaan dapat mengambil tindakan preventif yang tepat. Proyek ini menggunakan teknik klasifikasi untuk membangun model prediktif churn berdasarkan informasi historis pelanggan.

Salah satu studi dari [Verbraken et al., 2013](https://doi.org/10.1016/j.eswa.2012.08.009) menunjukkan bahwa metode ensemble seperti Random Forest efektif dalam prediksi churn dengan data yang tidak seimbang. Oleh karena itu, proyek ini mengimplementasikan beberapa algoritma termasuk Random Forest dan Neural Network.

## Business Understanding

### Problem Statements
- Bagaimana memprediksi apakah pelanggan akan melakukan churn berdasarkan data historis?
- Algoritma klasifikasi mana yang memberikan performa terbaik dalam memprediksi churn?

### Goals
- Membangun model klasifikasi churn pelanggan.
- Membandingkan performa Logistic Regression, Random Forest, dan Neural Network.
- Mengidentifikasi metrik performa terbaik: akurasi, F1-score, dan ROC-AUC.

### Solution Statements
- Melatih 3 model klasifikasi: Logistic Regression, Random Forest, dan Neural Network (Keras).
- Menggunakan teknik preprocessing seperti normalisasi dan encoding.
- Mengukur kinerja model dengan akurasi, F1-score, dan ROC AUC.
- Menyimpan model terbaik menggunakan `joblib`.

## Data Understanding

Dataset: [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Dataset ini berisi 7.043 baris data pelanggan dengan 21 fitur.

### Variabel penting:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`: Data demografis pelanggan.
- `tenure`, `PhoneService`, `InternetService`, dll.: Data layanan yang digunakan.
- `MonthlyCharges`, `TotalCharges`: Biaya layanan.
- `Churn`: Target label (Yes/No).

## Data Preparation

Tahapan preprocessing:
- Menghapus kolom yang tidak relevan seperti `customerID`.
- Menangani missing value (`TotalCharges`).
- Label encoding untuk kolom kategorikal.
- Normalisasi fitur numerik menggunakan `MinMaxScaler`.
- Split data: 80% training, 20% testing.

## Modeling

Tiga model digunakan:
1. **Logistic Regression**: Model baseline.
2. **Random Forest Classifier**: Model ensemble dengan performa kuat terhadap data tidak linear.
3. **Neural Network (Keras Sequential)**:
   - 3 layer dense dengan Dropout
   - Aktivasi `relu` dan `sigmoid`
   - Optimizer: Adam
   - Callback: EarlyStopping

## Evaluation

### Metrik evaluasi:
- **Accuracy**: Proporsi prediksi yang benar.
- **F1-Score**: Harmoni antara precision dan recall.
- **ROC-AUC**: Mengukur kemampuan klasifikasi biner dalam membedakan kelas.

### Hasil evaluasi:
- **Logistic Regression**
  - Akurasi: ~80%
  - F1-score: ~0.71
- **Random Forest**
  - Akurasi: ~82%
  - F1-score: ~0.73
- **Neural Network**
  - Akurasi: ~85%
  - F1-score: ~0.76
  - ROC-AUC: 0.89

Model Neural Network memberikan performa terbaik dan dipilih sebagai model akhir.

---

