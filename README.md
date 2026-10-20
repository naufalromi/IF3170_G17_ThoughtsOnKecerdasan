# Tugas Besar 1 IF3170 Inteligensi Artifisial - ThoughtsOnKecerdasan (2026)

## Anggota Kelompok

- 13524058 Muhammad Naufal Romi Annafi
- 13524083 Mahmudia Kimdaro Amin
- 13524129 Eduard Daniel Ariajaya

## Deskripsi

Program ini mencari susunan paket di dalam truk untuk memaksimalkan nilai paket yang diangkut. Susunan harus memenuhi batas ukuran truk, kapasitas berat, serta aturan tabrakan dan penopang paket. Pencarian dilakukan menggunakan algoritma *local search* melalui aplikasi GUI.

## Fitur

- Empat varian Hill Climbing: Steepest Ascent, Stochastic, Sideways Move, dan Random Restart.
- Simulated Annealing dan Genetic Algorithm.
- Pilihan fungsi objektif total value atau value dengan prioritas ETA.
- Input paket dan truk melalui GUI atau berkas JSON.
- Visualisasi state awal, state akhir, dan perkembangan nilai objektif.
- Penyimpanan hasil dalam format JSON dan PNG.

## Struktur Program

```text
.
├── app.py                 # Aplikasi GUI
├── src/                   # Model, aturan persoalan, algoritma, dan visualisasi
├── test/                  # Contoh input JSON
├── output/                # Hasil eksperimen
├── docs/                  # Dokumen laporan
├── requirements.txt       # Dependensi Python
└── README.md
```

## Prasyarat

- Python 3.10 atau lebih baru.
- pip untuk menginstal dependensi Python.
- Tkinter untuk menjalankan GUI.

## Cara Setup

Jalankan perintah berikut dari direktori utama repository:

```bash
python -m pip install -r requirements.txt
```

Di Windows, gunakan `python -m pip install -r requirements.txt`. Jika Tkinter belum tersedia, instal paket Tkinter yang sesuai dengan instalasi Python Anda.

## Cara Menjalankan Program

```bash
python app.py
```

1. Isi data truk dan paket, atau klik **Muat JSON** untuk memilih input seperti `test/tc1.json`.
2. Pilih algoritma, fungsi objektif, seed, dan parameter pencarian.
3. Klik **Jalankan**.
4. Buka tab **State awal**, **State akhir**, **Grafik**, atau **Rincian paket** untuk melihat hasil.
5. Klik **Simpan hasil (JSON + PNG)** dan pilih folder tujuan jika ingin menyimpan hasil.

## Pembagian Tugas

| Tugas | NIM |
| --- | --- |
| Steepest Ascent Hill-Climbing | 13524129 |
| Simulated Annealing | 13524083 |
| Genetic Algorithm | 13524058 |
| Integrasi | 13524058, 13524129, 13524083 |
| Core | 13524058 |
| Models | 13524058 |
| Hill-Climbing with Sideways Move | 13524129, 13524083 |
| Stochastic Hill-Climbing | 13524129 |
| Random Restart Hill-Climbing | 13524129, 13524083 |
| Objective Function Baru | 13524058 |