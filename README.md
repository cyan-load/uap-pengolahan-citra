# UAP Pengolahan Citra

## Klasifikasi Sampah Berdasarkan Konsep 3R Menggunakan Hybrid CNN dan GLCM

### Deskripsi

Proyek ini merupakan aplikasi klasifikasi sampah berbasis citra digital menggunakan metode Hybrid Convolutional Neural Network (CNN) dan Gray Level Co-occurrence Matrix (GLCM). Sistem mengklasifikasikan objek sampah ke dalam tiga kategori pengelolaan 3R, yaitu:

* Recycle
* Reuse
* Reduce

Aplikasi dikembangkan menggunakan Flask sebagai web interface sehingga pengguna dapat mengunggah gambar sampah dan memperoleh hasil klasifikasi secara langsung.

### Teknologi yang Digunakan

* Python
* Flask
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* Scikit-image
* NumPy
* Pandas

### Struktur Proyek

```text
dataset/
outputs/
src/
requirements.txt
README.md
```

### Cara Menjalankan

1. Clone repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi

```bash
python src/web/app.py
```

4. Buka browser

```text
http://127.0.0.1:5000
```

### Hasil

Model hybrid CNN dan GLCM berhasil melakukan klasifikasi sampah ke dalam kategori Recycle, Reuse, dan Reduce melalui antarmuka web berbasis Flask.

### 

1. Julianti Putri Azzahra - 065123126
2. Cahya Rahmatunnisa - 065123128
3. Amelia Setiana Wally - 065123129