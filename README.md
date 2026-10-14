# Museum AI Guide

Sistem **AI Guide Museum** merupakan aplikasi panduan museum berbasis kecerdasan buatan (*Artificial Intelligence/AI*) yang dirancang untuk membantu pengunjung memperoleh informasi mengenai artefak museum melalui interaksi tanya jawab.

Proyek ini mengintegrasikan backend Python, pendekatan *Retrieval-Augmented Generation* (RAG), model embedding BGE-M3, basis data vektor ChromaDB, model bahasa lokal melalui Ollama, serta Unity sebagai antarmuka interaksi pengguna.

Repository ini disiapkan agar kode dapat dipelajari, dikembangkan, dan digunakan kembali oleh mahasiswa atau pengembang lain.

## 1. Fitur Utama

- **AI Guide berbasis RAG:** menghasilkan jawaban dengan memanfaatkan informasi dari sumber data artefak yang tersedia.
- **Semantic Retrieval:** menggunakan BGE-M3 untuk membantu menemukan informasi yang relevan dengan pertanyaan pengunjung.
- **Vector Database:** menggunakan ChromaDB untuk menyimpan dan mengambil potongan informasi (*chunks*) hasil pengindeksan.
- **Local Language Model:** menggunakan Ollama untuk menjalankan model bahasa secara lokal.
- **Persona AI Guide:** mendukung konfigurasi persona untuk menyesuaikan perspektif penjelasan.
- **Integrasi Unity:** memungkinkan antarmuka Unity mengirim pertanyaan ke backend dan menampilkan jawaban yang diterima.
- **Konfigurasi yang dapat digunakan kembali:** memisahkan kode aplikasi dari model, database hasil pengindeksan, dan data lokal.

## 2. Arsitektur Sistem

Alur kerja sistem secara umum adalah sebagai berikut:

1. Pengunjung memilih artefak dan persona, kemudian memasukkan pertanyaan melalui Unity.
2. Unity mengirimkan permintaan ke backend Python melalui HTTP API.
3. Backend menggunakan BGE-M3 dan ChromaDB untuk mengambil konteks yang relevan berdasarkan artefak aktif.
4. Konteks hasil retrieval digunakan sebagai sumber informasi untuk model bahasa yang berjalan melalui Ollama.
5. Backend mengembalikan jawaban dalam respons API.
6. Unity menampilkan jawaban kepada pengunjung.

Komponen utama sistem:

| Komponen | Fungsi |
|---|---|
| Unity | Antarmuka pengguna dan interaksi dengan AI Guide |
| Python API | Menerima permintaan dan mengatur alur pemrosesan |
| BGE-M3 | Menghasilkan embedding untuk pencarian semantik |
| ChromaDB | Menyimpan dan mengambil vektor serta metadata dokumen |
| Ollama | Menjalankan model bahasa lokal |
| Qwen3 | Menghasilkan jawaban berdasarkan prompt dan konteks |
| Data artefak | Menyediakan informasi yang digunakan dalam retrieval |

## 3. Struktur Repository

Struktur repository dapat disesuaikan dengan implementasi aktual:

```text
museum-ai-guide/
├── backend/
│   ├── api_server.py
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── indexer.py
│   │   └── vector_store.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── unity/
│   └── Scripts/
│       └── AIGuideApiClient.cs
├── data/
│   └── README.md
├── .gitignore
└── README.md
```

Nama dan lokasi file di atas merupakan struktur acuan. Pastikan seluruh modul yang benar-benar digunakan oleh backend disertakan sebelum repository dibagikan.

## 4. Prasyarat

Sebelum menjalankan aplikasi, siapkan:

- Windows 10 atau Windows 11.
- Python dengan versi yang kompatibel dengan dependensi proyek.
- Ollama yang telah terpasang dan dapat dijalankan.
- Model bahasa Qwen3 yang digunakan oleh backend.
- Model embedding BGE-M3.
- Dependensi Python yang tercantum dalam `backend/requirements.txt`.
- Unity dengan proyek dan komponen UI AI Guide yang sesuai.
- Data artefak yang sah untuk digunakan serta prosedur untuk membangun indeks ChromaDB.

Kebutuhan perangkat keras bergantung pada ukuran model, konfigurasi Ollama, dan beban pemrosesan.

## 5. Mengunduh Repository

Clone repository menggunakan URL repository privat yang telah diberikan aksesnya:

```bash
git clone <URL_REPOSITORY>
cd museum-ai-guide
```

Ganti `<URL_REPOSITORY>` dengan URL repository GitHub yang sebenarnya.

Pengguna harus memiliki izin akses ke repository privat sebelum dapat melakukan clone.

## 6. Menyiapkan Backend Python

### 6.1. Membuat virtual environment

Jalankan perintah berikut dari direktori repository:

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

Jika PowerShell membatasi eksekusi skrip, gunakan metode aktivasi virtual environment yang sesuai dengan kebijakan keamanan perangkat.

### 6.2. Memasang dependensi

Pastikan `requirements.txt` berisi dependensi dan versi paket yang benar-benar dibutuhkan proyek.

Kemudian jalankan:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika instalasi gagal, periksa versi Python dan pesan kesalahan paket yang dilaporkan.

### 6.3. Menyiapkan konfigurasi

File `.env.example` menyediakan contoh konfigurasi yang dapat dijadikan acuan. Jika backend menggunakan environment variable, salin file tersebut menjadi `.env` dan sesuaikan nilainya dengan konfigurasi lokal.

Contoh konfigurasi:

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
CHAT_MODEL=qwen3:1.7b
EMBEDDING_MODEL=bge-m3
RAG_TOP_K=5
```

Nama variabel harus sesuai dengan yang benar-benar dibaca oleh kode backend. File `.env.example` tidak otomatis membuat konfigurasi tersebut aktif.

Jangan mengunggah `.env` atau informasi rahasia ke repository.

## 7. Menyiapkan Ollama dan Model

Pastikan Ollama telah terpasang dan berjalan. Untuk mengunduh model yang digunakan dalam konfigurasi awal, jalankan:

```powershell
ollama pull qwen3:1.7b
ollama pull bge-m3
```

Periksa model yang tersedia:

```powershell
ollama list
```

Nama model harus konsisten dengan konfigurasi backend. Model bahasa dan model embedding memiliki fungsi berbeda: Qwen3 digunakan untuk menghasilkan jawaban, sedangkan BGE-M3 digunakan untuk mendukung pencarian semantik.

Model diunduh melalui Ollama pada masing-masing komputer dan tidak disimpan sebagai berkas model di repository GitHub.

## 8. Menyiapkan Data dan ChromaDB

Database vektor lokal tidak harus disertakan dalam repository. Pengguna dapat membangun ulang indeks dari sumber data artefak yang telah diizinkan untuk digunakan.

Langkah umumnya:

1. Siapkan data artefak sesuai format yang dibaca oleh pipeline RAG.
2. Periksa struktur data, identitas artefak, dan metadata sumber.
3. Jalankan modul chunking untuk memecah dokumen menjadi potongan informasi.
4. Jalankan indexer untuk membuat embedding dan menyimpan dokumen ke ChromaDB.
5. Verifikasi jumlah dokumen, metadata `artifact_id`, dan hasil retrieval.

Contoh perintah berikut berlaku jika modul tersebut dapat dijalankan sebagai modul Python dari direktori `backend`:

```powershell
python -m rag.chunking
python -m rag.indexer
```

Periksa implementasi masing-masing modul terlebih dahulu karena perintah, argumen, lokasi data, dan lokasi database dapat berbeda sesuai konfigurasi proyek.

**Catatan:** data artefak, hasil wawancara, dokumentasi internal, dan database yang tidak diizinkan untuk dibagikan tidak boleh dimasukkan ke repository. Pengguna harus memperoleh data melalui sumber yang sah.

## 9. Menjalankan Backend API

Pastikan Ollama aktif, dependensi telah terpasang, data telah disiapkan, dan indeks ChromaDB tersedia.

Jalankan API menggunakan metode startup yang ditetapkan dalam `api_server.py` atau dokumentasi backend.

Contoh berikut hanya berlaku jika aplikasi FastAPI diekspos melalui objek `app` dalam modul `api_server.py`:

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8001
```

Jika objek aplikasi, nama modul, atau port berbeda, sesuaikan perintah dengan implementasi aktual.

Setelah server berjalan, uji endpoint kesehatan jika tersedia:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Pastikan endpoint dan responsnya sesuai dengan yang didefinisikan dalam kode backend.

## 10. Menghubungkan Unity

Unity berfungsi sebagai klien yang berkomunikasi dengan backend melalui HTTP API.

Secara umum, integrasi membutuhkan:

- Input pertanyaan pengunjung.
- Pemilihan artefak.
- Pemilihan persona.
- Pengiriman permintaan JSON ke endpoint chat.
- Penanganan respons dan error.
- Penampilan jawaban pada panel AI Guide.

Contoh konfigurasi URL API untuk pengujian pada komputer yang sama:

```text
http://127.0.0.1:8001/api/chat
```

Pastikan port sesuai dengan server Python yang sedang berjalan.

Jika Unity dijalankan pada komputer berbeda, alamat loopback `127.0.0.1` harus diganti dengan alamat host backend yang dapat dijangkau dari komputer tersebut.

Nilai `artifact_id`, `persona`, `gender`, dan `museum_id` yang dikirim oleh Unity harus sesuai dengan skema permintaan backend.

## 11. Pengujian Sistem

Lakukan pengujian secara bertahap:

1. Pastikan API dapat dijalankan tanpa error.
2. Pastikan Ollama dapat mengakses model yang dibutuhkan.
3. Pastikan ChromaDB memiliki dokumen untuk artefak yang diuji.
4. Kirim pertanyaan melalui endpoint chat.
5. Periksa apakah `artifact_id` sesuai dengan artefak aktif.
6. Periksa apakah retrieved chunks relevan dengan pertanyaan.
7. Periksa konsistensi nama, tahun, jumlah, dan komponen artefak.
8. Bandingkan perspektif jawaban berdasarkan persona yang dipilih.
9. Pastikan jawaban mengikuti bahasa pertanyaan dan tidak mengarang fakta.
10. Uji pengiriman pertanyaan dan penampilan jawaban melalui Unity.

Informasi yang tidak tersedia dalam sumber harus dinyatakan sebagai informasi yang belum ditemukan atau belum terkonfirmasi sesuai konteks sumber.

## 12. Keamanan dan Pengelolaan Repository

Sebelum melakukan commit atau push, pastikan repository tidak berisi:

- File `.env` dan kredensial.
- Model AI berukuran besar.
- Folder virtual environment.
- Database ChromaDB lokal.
- Cache dan file log.
- Data museum atau wawancara yang belum memperoleh izin untuk dibagikan.
- File lokal yang tidak diperlukan untuk menjalankan sistem.

Gunakan `.gitignore` dan lakukan pemeriksaan perubahan yang akan di-commit.

Repository privat membatasi akses kepada pengguna yang diberi izin, tetapi tidak menggantikan kewajiban untuk melindungi data sensitif.

## 13. Pengembangan Lebih Lanjut

Pengembangan berikutnya dapat mencakup:

- Penyempurnaan prompt persona.
- Peningkatan kualitas retrieval.
- Evaluasi konsistensi dan kelengkapan jawaban.
- Pemetaan artefak dari label UI ke `artifact_id`.
- Penanganan timeout dan error komunikasi.
- Optimasi performa backend.
- Dokumentasi pengujian dan reproduksibilitas hasil.

Setiap perubahan sebaiknya diuji pada backend terlebih dahulu sebelum diintegrasikan ke Unity.

## 14. Lisensi dan Penggunaan

Repository ini ditujukan untuk pengembangan, pembelajaran, dan penggunaan ulang kode AI Guide museum oleh pihak yang memperoleh akses.

Ketentuan penggunaan, distribusi, dan modifikasi kode perlu mengikuti lisensi yang ditetapkan oleh pemilik repository. Hak penggunaan data artefak, foto, wawancara, serta model pihak ketiga mengikuti ketentuan masing-masing sumber.

---

**Museum AI Guide**  
Backend Python · Ollama · RAG · BGE-M3 · ChromaDB · Unity
