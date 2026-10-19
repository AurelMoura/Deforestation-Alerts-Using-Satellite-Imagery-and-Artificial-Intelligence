# Deforestation-Alerts-Using-Satellite-Imagery-and-Artificial-Intelligence
# Klasifikasi Gangguan Tutupan Hutan Menggunakan Machine Learning

## Deskripsi Proyek

Proyek ini dibuat sebagai studi kasus kegiatan Informatics Practitioner Lecture Series 2026 pada materi *Deforestation Alerts Using Satellite Imagery and Artificial Intelligence* yang disampaikan oleh Dr. Bart Slater dari Wageningen University & Research, Belanda.

Proyek ini menerapkan konsep dasar Machine Learning untuk mengklasifikasikan kategori gangguan tutupan hutan berdasarkan fitur simulasi.

## Tujuan

- Memahami penerapan Machine Learning untuk klasifikasi lingkungan.
- Mempelajari tahapan persiapan data, pelatihan model, dan evaluasi.
- Membandingkan hasil klasifikasi menggunakan accuracy dan macro F1-score.
- Memahami pentingnya kualitas data dalam pengembangan model AI.

## Teknologi

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Algoritma Random Forest

## Kategori Klasifikasi

Model mempelajari enam kategori dari dataset simulasi:

1. Pertanian Skala Besar
2. Pertanian Skala Kecil
3. Pembangunan Jalan
4. Penebangan Selektif
5. Pertambangan
6. Kebakaran
7. Gangguan Alami

## Fitur Dataset

Dataset `data_hutan.csv` memiliki empat fitur numerik, yaitu perubahan vegetasi, perubahan kelembapan, akses jalan, dan aktivitas pembukaan. Kolom `kategori` menjadi label yang dipelajari oleh model.

## Cara Menjalankan

1. Pastikan Python sudah terpasang.
2. Buka terminal pada folder proyek.
3. Instal pustaka yang dibutuhkan:

   ```bash
   python -m pip install pandas matplotlib scikit-learn
   ```

4. Jalankan program:

   ```bash
   python klasifikasi_hutan.py
   ```

5. Periksa hasil evaluasi di terminal dan buka file `hasil_evaluasi.png`.

## Evaluasi Model

Program menampilkan accuracy, macro F1-score, classification report, dan confusion matrix. Macro F1-score membantu melihat performa model pada setiap kelas tanpa membiarkan kelas yang jumlahnya lebih besar mendominasi skor rata-rata.

## Keterbatasan

Dataset yang digunakan adalah data simulasi untuk pembelajaran, bukan citra satelit Sentinel-1 atau Sentinel-2. Model ini menggunakan Random Forest, bukan CNN, dan belum divalidasi menggunakan data lapangan. Oleh karena itu, hasilnya tidak boleh dianggap sebagai bukti deteksi deforestasi nyata.

## Kesimpulan

Proyek ini memperlihatkan alur dasar klasifikasi gangguan hutan menggunakan Machine Learning, mulai dari pembacaan data sampai evaluasi model. Proyek ini juga menunjukkan bahwa hasil klasifikasi perlu dinilai dengan metrik yang sesuai dan bahwa model AI sangat bergantung pada kualitas serta keterwakilan data pelatihan.
