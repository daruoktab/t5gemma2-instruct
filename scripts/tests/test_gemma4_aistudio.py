import os
import glob
from google import genai

client = genai.Client()

# Temukan semua gambar halaman di scratch/output_images/
image_dir = os.path.join(os.path.dirname(__file__), "output_images")
image_files = sorted(
    glob.glob(os.path.join(image_dir, "page_*.png")),
    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
)

print(f"Menemukan {len(image_files)} gambar halaman untuk diunggah.")

# 1. Unggah semua file gambar ke GenAI Files API
uploaded_files = []
for img_path in image_files:
    print(f"Mengunggah ke Files API: {img_path}")
    uploaded = client.files.upload(file=img_path)
    uploaded_files.append(uploaded)

# 2. Susun input dengan struktur yang benar untuk client.interactions.create
contents = [
    {
        "type": "text",
        "text": "Berikut adalah halaman-halaman dari paper ilmiah. Tolong jelaskan dan deskripsikan secara ringkas poin-poin penting dari setiap halaman ini (Halaman 1 sampai 13) secara berurutan."
    }
]

for uploaded in uploaded_files:
    contents.append({
        "type": "image",
        "uri": uploaded.uri,
        "mime_type": uploaded.mime_type
    })

print("\nMengirim data ke model gemma-4-31b-it via Interactions API...")
interaction = client.interactions.create(
    model="gemma-4-31b-it",
    input=contents
)

print("\n=== Hasil Deskripsi Model ===")
print(interaction.output_text)  # type: ignore

# 3. Bersihkan file yang diunggah (opsional, tapi disarankan)
print("\nMembersihkan file dari Files API...")
for uploaded in uploaded_files:
    if uploaded.name:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"Gagal menghapus file {uploaded.name}: {e}")