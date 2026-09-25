"""Builds the knowledge-base text and the CS-agent prompt for one message."""

INTENTS = ("harga", "stok", "ongkir", "order", "komplain", "info")


def _rupiah(n: int) -> str:
    return "Rp " + f"{n:,}".replace(",", ".")


def render_kb(store: dict) -> str:
    lines = [f"TOKO: {store['name']} ({store['location']})", "", "KATALOG:"]
    lines.append(store["catalog_note"])
    for p in store["catalog"]:
        stock = f"stok {p['stock']}" if p["stock"] > 0 else "stok HABIS (0)"
        lines.append(f"- [{p['sku']}] {p['name']} | {_rupiah(p['price'])} | {stock}")
    lines += ["", "ONGKIR (dari " + store["location"] + "):"]
    for s in store["shipping"]:
        lines.append(f"- {s['city']}: {_rupiah(s['cost'])}, estimasi {s['eta']}")
    lines.append(store["shipping_note"])
    lines += ["", "KEBIJAKAN TOKO:"]
    lines += [f"- {p}" for p in store["policies"]]
    return "\n".join(lines)


GUARD_TEXT = """Jawab HANYA berdasarkan BASIS PENGETAHUAN di bawah. Jika informasi yang ditanyakan pelanggan
tidak ada di basis pengetahuan, JANGAN menebak atau mengarang angka/fakta: sampaikan dengan
sopan bahwa admin akan mengecek dulu."""

# Ablation: same prompt with the anti-guessing instruction replaced by a neutral line.
NO_GUARD_TEXT = "Gunakan BASIS PENGETAHUAN di bawah untuk membantu pelanggan."

PROMPT_TEMPLATE = """Kamu adalah admin customer service WhatsApp untuk toko online "{name}".
{guard}

=== BASIS PENGETAHUAN ===
{kb}
=== AKHIR BASIS PENGETAHUAN ===

Pesan WhatsApp dari pelanggan (bisa terdiri dari beberapa bubble chat):
<<<
{message}
>>>

Balas dengan SATU objek JSON saja (tanpa teks lain, tanpa markdown) dengan field berikut:
- "intent": salah satu dari "harga" (tanya harga), "stok" (tanya ketersediaan), "ongkir" (tanya ongkos/waktu/area kirim), "order" (ingin memesan), "komplain" (masalah dengan pesanan yang sudah diterima), "info" (hal lain: pembayaran, jam buka, kebijakan, detail produk)
- "sku": kode SKU produk yang dimaksud pelanggan persis seperti di katalog (contoh "{example_sku}"), atau null jika tidak menyebut produk spesifik
- "quantity": jumlah barang yang disebut pelanggan sebagai integer, atau null jika tidak disebut
- "city": nama resmi kota tujuan/lokasi pelanggan (contoh "Yogyakarta", "Surabaya"), atau null jika tidak disebut
- "can_answer_from_kb": true jika pertanyaan utama pelanggan bisa dijawab tuntas dari basis pengetahuan, false jika tidak
- "reply": balasan untuk pelanggan dalam Bahasa Indonesia, ramah dan singkat"""


def build_prompt(store: dict, message: str, guard: bool = True) -> str:
    return PROMPT_TEMPLATE.format(
        name=store["name"],
        guard=GUARD_TEXT if guard else NO_GUARD_TEXT,
        kb=render_kb(store),
        message=message,
        example_sku=store["catalog"][0]["sku"],
    )
