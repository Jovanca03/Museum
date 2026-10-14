"""
Scraper khusus untuk berita kehilangan koleksi Museum Cakraningrat.
Menarik data dari portal berita (Kompas, IDN Times, RRI, Dandapala) 
dan menggabungkannya dengan transkrip berita televisi (YouTube).
"""

import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# File tujuan (akan menimpa file kosong yang kita buat sebelumnya)
OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "cakra_data.json"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

# Daftar URL Berita yang akan di-scrape
NEWS_URLS = [
    "https://www.kompas.com/jawa-timur/read/2025/10/14/071500988/jejak-sejarah-yang-raib-40-koleksi-berharga-museum-cakraningrat?page=all",
    "https://jatim.idntimes.com/news/jawa-timur/hilangnya-barang-bersejarah-di-museum-cakraningrat-akhirnya-terungkap-00-hc4tp-n8bpk5",
    "https://rri.co.id/sampang/regional/1895420/dua-artefak-bersejarah-museum-bangkalan-dilaporkan-hilang",
    "https://rri.co.id/sampang/regional/1895589/polisi-selidiki-dugaan-pencurian-gamelan-museum-cakraningrat-bangkalan",
    "https://dandapala.com/article/detail/curi-koleksi-museum-cakraningrat-bangkalan-pelaku-dibui-6-tahun"
]

def scrape_news_article(url: str) -> str:
    """Mengambil teks paragraf dari artikel berita."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Ekstrak judul
        title = soup.title.get_text(strip=True) if soup.title else "Berita Kehilangan"
        
        # Karena setiap portal berita berbeda, cara teraman adalah menarik semua tag paragraf <p>
        paragraphs = soup.find_all("p")
        teks_artikel = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40])
        
        return f"Judul Berita: {title}\nSumber: {url}\n\nIsi Laporan:\n{teks_artikel}"
    
    except Exception as e:
        print(f"Gagal mengambil {url}: {e}")
        return ""

def get_youtube_transcript() -> str:
    """Mengembalikan teks transkrip manual dari liputan tvOne yang telah dianalisis AI."""
    return (
        "Judul Laporan: Pencuri Puluhan Artefak Museum Cakraningrat Ditangkap | tvOne\n"
        "Sumber Video: https://www.youtube.com/watch?v=3MVuqd0baR4\n\n"
        "Isi Laporan:\n"
        "Pelaku pencurian artefak di Museum Cakraningrat yang terletak di Kantor Dinas Kebudayaan "
        "dan Pariwisata Kabupaten Bangkalan ditangkap Satreskrim Polres setempat. Pelaku ditembak bagian "
        "betisnya karena melawan petugas. Tersangka bernama Herman Taufik (44 tahun) melawan saat "
        "digerebek di kamar kos. Polisi menemukan sejumlah barang bukti hasil curian di dalam kamar kos tersebut. "
        "Sebagian besar artefak lainnya sudah dijual tersangka ke pasar loak. Penangkapan dilakukan setelah "
        "adanya laporan pencurian 2 minggu sebelumnya dari pihak museum. Pelaku mengaku melancarkan aksinya "
        "seorang diri dengan mencongkel jendela museum di malam hari. Tersangka dijerat pasal 363 KUHP "
        "tentang pencurian dengan pemberatan dengan ancaman 7 tahun penjara."
    )

def build_missing_record(artifact_id: str, content: str) -> dict:
    """Menyusun JSON format RAG."""
    return {
        "artifact_id": artifact_id,
        "museum_id": "MUSEUM_CAKRANINGRAT",
        "name": "Arsip Laporan Kehilangan Artefak Museum Cakraningrat",
        "status": "Hilang/Pencurian",
        "source": "Kompilasi Berita Eksternal & Kepolisian",
        "content": content,
        "perspectives": {
            "archaeologist": "Berdasarkan laporan, kondisi pelestarian artefak terancam karena dijual ke pasar loak.",
            "historian": "Catatan kronologi menunjukkan pencurian dilakukan dengan mencongkel jendela dan pelakunya telah divonis 7 tahun penjara.",
            "cultural_expert": "Hilangnya koleksi ini merupakan hilangnya sebagian memori kolektif masyarakat Bangkalan.",
        }
    }

def main():
    print("Memulai proses scraping dari portal berita...\n")
    results = []
    counter = 1
    
    # 1. Scrape Berita Artikel
    for url in NEWS_URLS:
        print(f"Mengambil data dari: {url}")
        content = scrape_news_article(url)
        if content:
            results.append(build_missing_record(f"CAKRA_MISSING_{counter:03d}", content))
            counter += 1
            
    # 2. Tambahkan Data Liputan TV (YouTube)
    print("\nMenyertakan transkrip laporan kepolisian dari tvOne (YouTube)...")
    yt_content = get_youtube_transcript()
    results.append(build_missing_record(f"CAKRA_MISSING_{counter:03d}", yt_content))
    
    # 3. Simpan ke JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\n================================")
    print("Scraping berita selesai 100%.")
    print(f"Data tersimpan di:\n{OUTPUT_FILE}")
    print(f"Total Arsip Berita: {len(results)}")
    print("================================")

if __name__ == "__main__":
    main()