# Prediksi Biaya Asuransi Kesehatan dengan Machine Learning

Halo, saya **Raihan Putra Nurhamsyah**, mahasiswa Teknik Informatika Universitas Pamulang. Selamat datang di repositori tugas praktikum machine learning saya! 👋

Di repositori ini saya membangun model untuk **memprediksi biaya asuransi kesehatan** seseorang berdasarkan data pribadinya, seperti usia, BMI, status perokok, dan wilayah tempat tinggal. Seluruh prosesnya saya kerjakan dari nol sampai model jadi, mengikuti alur yang diajarkan dosen saya: mulai dari mengumpulkan data, membersihkan, menganalisis, membagi data, melatih model, sampai mengevaluasi dan menyimpannya.

README ini saya tulis supaya siapa pun yang membuka repositori ini, bahkan yang belum pernah belajar machine learning, bisa paham **apa yang saya kerjakan dan kenapa setiap langkahnya perlu dilakukan**.

---

## Sebenarnya, ini proyek tentang apa?

Bayangkan sebuah perusahaan asuransi. Setiap calon nasabah punya kondisi yang berbeda-beda, dan biaya premi atau klaimnya pun berbeda. Pertanyaannya: **faktor apa yang paling memengaruhi biaya itu, dan bisakah kita memprediksinya?**

Itulah yang saya jawab di proyek ini. Karena yang diprediksi adalah angka (biaya dalam dolar), jenis masalahnya adalah **regresi** pada *supervised learning*. Kalau targetnya berupa kategori, misalnya kelas biaya "rendah, sedang, tinggi", maka algoritma yang dipakai berganti menjadi klasifikasi.

## Dataset yang dipakai

Saya memakai **Medical Cost Personal Datasets** (`insurance.csv`), dataset terbuka yang tersedia di Kaggle dan GitHub. Isinya 1.338 baris data dengan 7 kolom:

| Kolom | Penjelasan |
|---|---|
| `age` | Usia penerima manfaat utama |
| `sex` | Jenis kelamin (female atau male) |
| `bmi` | Body mass index, ukuran proporsi berat dan tinggi badan |
| `children` | Jumlah anak atau tanggungan yang dicakup asuransi |
| `smoker` | Apakah perokok atau bukan (yes atau no) |
| `region` | Wilayah tinggal di AS (northeast, northwest, southeast, southwest) |
| `charges` | **Biaya asuransi, ini target yang diprediksi** |

---

## Perjalanan saya, tahap demi tahap

Berikut urutan yang saya kerjakan di notebook. Di setiap tahap saya jelaskan **apa yang dilakukan** dan **kenapa harus dilakukan**.

### 1. Data Collecting dan Data Loading
Saya mengambil dataset dari sumber terbuka (lewat tautan GitHub) menggunakan `pandas`, dengan file lokal sebagai cadangan jika internet bermasalah. Setelah itu saya membuat salinan data kerja `df`.

**Kenapa begitu?** Semua proyek machine learning dimulai dari data. Salinan dibuat supaya data asli tetap utuh. Kalau suatu langkah pembersihan salah, saya tinggal mengulang dari data asli.

### 2. Data Cleaning (membersihkan data)
Data mentah hampir selalu "kotor", dan model yang dilatih dari data kotor akan memberi hasil yang kurang bisa dipercaya. Ada empat hal yang saya periksa:

- **Informasi dan statistik dataset** (`info()` dan `describe()`): memastikan tipe data benar dan melihat sebaran awal angka. Dari sini terlihat biaya (`charges`) sangat condong ke kanan, artinya sebagian besar biaya kecil dan hanya sedikit yang sangat besar.
- **Missing value** (data kosong): saya memeriksa dan menyiapkan penanganannya (numerik diisi median, kategorikal diisi modus, kolom yang kosongnya lebih dari 75% dibuang). Pada dataset ini ternyata **tidak ada data kosong**, tetapi kodenya tetap saya jalankan agar alurnya lengkap dan bisa dipakai ulang di dataset lain.
- **Outlier** (nilai ekstrem): saya mendeteksi dengan box plot dan metode IQR. Hasilnya, 9 baris dengan BMI sangat tinggi saya hapus. **Saya sengaja tidak menghapus outlier pada kolom `charges`**, karena biaya yang tinggi itu data asli milik perokok, bukan kesalahan input. Kalau dihapus (sekitar 10% data), model akan kehilangan pola terpenting yang justru harus ia pelajari.
- **Data duplikat**: ada 1 baris ganda yang saya hapus supaya satu pengamatan tidak dihitung dua kali.

### 3. Data Transformation (mengubah data agar siap dipakai model)
- **Standardisasi** (`StandardScaler`): fitur numerik seperti `age`, `bmi`, dan `children` diubah agar rata-ratanya 0 dan simpangan bakunya 1. *Kenapa?* Skala antarfitur berbeda jauh (usia puluhan, jumlah anak satuan), dan model regresi lebih stabil bila skalanya seragam. Kolom target `charges` sengaja tidak diskalakan supaya nilai error tetap dalam dolar dan mudah dipahami.
- **Encoding**: komputer hanya mengerti angka, jadi kolom teks harus diubah. Saya membandingkan dua cara, yaitu *One-Hot Encoding* dan *Label Encoding*. Hasil akhirnya: `sex` dan `smoker` (hanya dua pilihan) memakai Label Encoding, sedangkan `region` (empat wilayah tanpa urutan) memakai One-Hot Encoding. *Kenapa dibedakan?* Kalau wilayah diberi angka 0, 1, 2, 3, model bisa salah mengira ada "urutan" antarwilayah.

Hasil akhir tahap ini adalah dataset final berukuran **1.328 baris × 10 kolom**.

### 4. EDA dan ExDA (mengenali data lewat grafik)
Sebelum melatih model, saya "berkenalan" dulu dengan data lewat visualisasi: distribusi biaya, perbandingan biaya antarwilayah, hubungan biaya dengan usia dan BMI, serta matriks korelasi.

**Kenapa penting?** Dari tahap ini saya sudah bisa menebak jawabannya sebelum model dibuat: **status perokok adalah faktor yang paling berpengaruh** (korelasi sekitar 0,79 terhadap biaya), disusul usia dan BMI. Ada juga temuan menarik: pada perokok, biaya naik tajam ketika BMI naik, sedangkan pada bukan perokok hampir tidak berpengaruh. Artinya ada *interaksi* antarfitur yang nanti menentukan model mana yang cocok.

### 5. Data Splitting (membagi data)
Data dibagi **80% untuk latih** (1.062 baris) dan **20% untuk uji** (266 baris) dengan `train_test_split`.

**Kenapa harus dibagi?** Ibarat belajar untuk ujian: kalau soal ujian sama persis dengan soal latihan, nilai bagus belum tentu berarti paham. Data uji adalah "soal ujian" yang belum pernah dilihat model, sehingga kemampuan model yang sebenarnya bisa dinilai secara jujur.

### 6. Modelling (membangun model)
Saya fokus pada **dua model** agar perbandingannya jelas:

- **Linear Regression**: model sederhana yang mencari garis lurus antara fitur dan biaya. Ia berperan sebagai *baseline* (pembanding dasar).
- **Gradient Boosting Regressor**: model yang membangun banyak pohon keputusan secara berurutan, di mana setiap pohon memperbaiki kesalahan pohon sebelumnya. Model ini mampu menangkap hubungan yang tidak lurus.

**Kenapa dua model?** Dengan membandingkan model sederhana dan model kompleks, kita bisa melihat apakah kerumitan tambahan itu benar-benar memberi hasil yang lebih baik.

### 7. Evaluasi Model
Kedua model diuji pada data uji dengan empat ukuran: **MAE** (rata-rata selisih prediksi dan nilai sebenarnya), **MSE**, **RMSE** (satuannya dolar), dan **R²** (seberapa besar variasi biaya yang bisa dijelaskan model, makin dekat ke 1 makin baik).

| Model | MAE | RMSE | R² (data uji) | R² (data latih) |
|---|---|---|---|---|
| **Gradient Boosting** | 2.559 | 4.510 | **0,876** | 0,899 |
| Linear Regression | 4.355 | 6.105 | 0,772 | 0,742 |

**Cara membacanya:** Gradient Boosting jauh lebih baik, dengan rata-rata kesalahan hampir separuh dari Linear Regression. R² data latih dan data ujinya juga berdekatan, jadi model tidak sekadar menghafal (tidak *overfitting*). Linear Regression kalah karena hubungan antara fitur dan biaya memang tidak lurus, terutama efek perokok yang berinteraksi dengan BMI dan usia.

Saya juga menambahkan grafik *feature importance* dan *plot residual*. Menurut Gradient Boosting, `smoker` menyumbang sekitar 68% pengaruh, `bmi` sekitar 18%, dan `age` sekitar 12%. Hasil ini konsisten dengan temuan di tahap EDA.

### 8. Penyimpanan Data dan Model
Dataset bersih dan model terbaik disimpan memakai `joblib` agar bisa dipakai lagi tanpa mengulang semua proses. `scaler` dan urutan kolom fitur juga ikut disimpan, karena data baru harus diproses dengan cara yang **persis sama** seperti saat model dilatih.

---

## Kesimpulan

Faktor yang paling memengaruhi biaya asuransi pada data ini adalah **status perokok**, lalu **usia** dan **BMI**. Model terbaik adalah **Gradient Boosting Regressor** dengan R² sekitar **0,88** pada data uji. Pelajaran terpentingnya: sebelum memilih model, pahami dulu datanya, karena pola yang saya temukan lewat EDA (efek perokok dan interaksinya dengan BMI) langsung menjelaskan kenapa model non-linear lebih unggul.

---

## Isi repositori

```
.
├── prediksi-biaya-asuransi.ipynb   # notebook utama (kode + penjelasan tiap langkah)
├── README.md                       # dokumen ini
└── (hasil setelah notebook dijalankan)
    ├── insurance_clean.csv         # dataset final hasil pemrosesan
    ├── gbr_model.joblib            # model Gradient Boosting yang sudah dilatih
    ├── scaler.joblib               # scaler untuk standardisasi data baru
    └── fitur_kolom.joblib          # urutan kolom fitur
```

Setiap cell kode di notebook diawali penjelasan singkat dalam bahasa Indonesia, jadi bisa dibaca seperti modul belajar.

## Cara menjalankan

**Opsi 1: Google Colab (paling mudah).** Unggah file `prediksi-biaya-asuransi.ipynb` ke [Google Colab](https://colab.research.google.com), lalu pilih *Runtime → Run all*. Data akan diunduh otomatis dari internet.

**Opsi 2: di komputer sendiri.**

```bash
# 1. Salin repositori
git clone <URL-REPOSITORI-INI>
cd <NAMA-FOLDER>

# 2. Pasang pustaka yang dibutuhkan
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Buka notebook
jupyter notebook prediksi-biaya-asuransi.ipynb
```

## Teknologi yang dipakai

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, dan joblib.

## Catatan dan rencana pengembangan

Saya menulis ini apa adanya supaya jujur soal batasannya:

- Standardisasi saat ini dilakukan **sebelum** pembagian data, mengikuti urutan materi dosen. Secara teori ini memungkinkan sedikit kebocoran informasi dari data uji. Perbaikan idealnya: `scaler` dilatih hanya pada data latih, misalnya memakai `Pipeline`.
- Evaluasi masih memakai satu kali pembagian data. Ke depan saya ingin mencoba *cross validation* dan *hyperparameter tuning*.
- Dataset ini kecil dan tidak memuat faktor seperti riwayat penyakit, sehingga masih ada kasus yang prediksinya meleset jauh.

## Sumber data

Choi, M. *Medical Cost Personal Datasets*, Kaggle: <https://www.kaggle.com/datasets/mirichoi0218/insurance> (data asli berasal dari buku *Machine Learning with R* karya Brett Lantz).

---

Terima kasih sudah mampir dan membaca sampai akhir! Kalau ada saran atau masukan, silakan buka *issue* di repositori ini. 🙌

**Raihan Putra Nurhamsyah**
Teknik Informatika, Universitas Pamulang
