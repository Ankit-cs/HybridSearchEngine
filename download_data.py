import urllib.request
import bz2
import os

URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
RAW_DIR = r"c:\astraverse\HybridSearchEngine\data\raw"
BZ2_FILE = os.path.join(RAW_DIR, "simplewiki.xml.bz2")
XML_FILE = os.path.join(RAW_DIR, "simplewiki.xml")

os.makedirs(RAW_DIR, exist_ok=True)

print(f"Downloading {URL}...")
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AstraSearch/1.0'})
with urllib.request.urlopen(req) as response, open(BZ2_FILE, 'wb') as out_file:
    out_file.write(response.read())
print("Download complete.")

print("Extracting...")
with bz2.open(BZ2_FILE, "rb") as f_in:
    with open(XML_FILE, "wb") as f_out:
        for data in iter(lambda: f_in.read(100 * 1024 * 1024), b''):
            f_out.write(data)
print("Extraction complete.")

os.remove(BZ2_FILE)
print("Done!")
