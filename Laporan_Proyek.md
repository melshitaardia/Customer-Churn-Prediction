
# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Domain Proyek: Telekomunikasi – Prediksi Customer Churn

Permasalahan customer churn atau pelanggan berhenti berlangganan adalah isu besar dalam industri telekomunikasi. Perusahaan kehilangan pelanggan secara langsung mempengaruhi pendapatan, dan biaya untuk menarik pelanggan baru jauh lebih mahal dibanding mempertahankan pelanggan lama. Oleh karena itu, prediksi customer churn menggunakan machine learning menjadi solusi strategis untuk membantu perusahaan memitigasi kehilangan pelanggan lebih awal.

**Referensi:**
- Blattberg, R. C., Getz, G., & Thomas, J. S. (2001). *Customer Equity: Building and Managing Relationships as Valuable Assets*. Harvard Business Press.

---

## Business Understanding

### Problem Statements:
1. Bagaimana cara mengidentifikasi pelanggan yang berpotensi churn (berhenti berlangganan)?
2. Fitur-fitur apa saja yang paling memengaruhi keputusan pelanggan untuk churn?
3. Model klasifikasi mana yang paling akurat untuk prediksi churn?

### Goals:
1. Membangun model klasifikasi churn pelanggan berdasarkan data historis.
2. Mengidentifikasi fitur-fitur penting yang berkontribusi terhadap keputusan churn.
3. Membandingkan beberapa model machine learning untuk menentukan model terbaik.

### Solution Statement:
- Membangun dan membandingkan 3+ model klasifikasi: Logistic Regression, Random Forest, SVM, KNN, dan XGBoost.
- Melakukan evaluasi model menggunakan metrik AUC, recall, precision, dan F1-score.
- Menggunakan SHAP dan feature importance untuk interpretabilitas model.

---

## Data Understanding

Dataset berasal dari Kaggle dengan nama "Telco Customer Churn Dataset":
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset ini berisi informasi pelanggan seperti jenis layanan yang digunakan, durasi berlangganan, tagihan bulanan, total pembayaran, dan status churn.

Contoh variabel:
- `gender`: jenis kelamin pelanggan
- `SeniorCitizen`: apakah pelanggan adalah warga senior
- `tenure`: lama berlangganan (dalam bulan)
- `MonthlyCharges`: tagihan bulanan
- `TotalCharges`: total biaya yang dibayarkan
- `Churn`: target variabel (1 = churn, 0 = tidak)

Dataset berisi lebih dari 7.000 sampel data (memenuhi syarat kuantitatif).

EDA juga dilakukan menggunakan visualisasi distribusi churn dan korelasi fitur numerik terhadap churn.

---

## Data Preparation

- Mengubah `TotalCharges` dari string ke float, lalu menangani missing values dengan menghapus baris terkait.
- Menghapus kolom tidak informatif (`customerID`).
- Melakukan one-hot encoding pada fitur kategorikal.
- Normalisasi fitur numerik menggunakan `MinMaxScaler`.
- Memisahkan dataset menjadi `X` dan `y`, lalu melakukan train-test split 80:20 dengan `stratify`.

Langkah-langkah ini bertujuan memastikan data bersih, dapat diterima model machine learning, dan mencegah bias karena skala fitur berbeda.

---

## Modeling

Model yang diuji:
- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost

Setiap model dilatih dengan parameter default terlebih dahulu, lalu dievaluasi dengan data test set.

Random Forest dipilih sebagai model terbaik karena:
- Konsisten memberikan AUC tinggi (~0.85)
- Memiliki interpretabilitas tinggi (feature importance)
- Lebih stabil pada data yang heterogen

---

## Evaluation

### Metrik Evaluasi:
- **Accuracy**: persentase prediksi benar dari total sampel.
- **Precision**: proporsi prediksi churn yang benar dari seluruh prediksi churn.
- **Recall (Sensitivity)**: proporsi churn yang berhasil diprediksi benar.
- **F1-score**: harmonisasi precision dan recall.
- **ROC AUC Score**: area under the curve pada ROC, mencerminkan kemampuan model membedakan antara kelas churn dan tidak.

### Hasil Evaluasi:
- ROC AUC tertinggi diraih oleh Random Forest (~0.85)
- Confusion matrix menunjukkan model cukup baik menangkap churn (positif class)
- SHAP digunakan untuk interpretasi fitur penting (misalnya, `tenure`, `MonthlyCharges`)

---

## Kesimpulan

Model klasifikasi berbasis Random Forest terbukti menjadi solusi terbaik untuk prediksi churn dalam dataset Telco. Kombinasi preprocessing yang tepat, evaluasi model yang menyeluruh, dan interpretasi model menjadikan solusi ini siap diterapkan pada bisnis nyata.

---

*Dokumen ini merupakan bagian dari submission kelas Machine Learning Terapan di Dicoding.*
