from __future__ import annotations

from detection.utils import read_json, write_json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str], pad_id: int, unk_id: int):
        self.stoi = stoi
        self.itos = itos
        self.pad_id = pad_id
        self.unk_id = unk_id

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        vocab = sorted(set("".join(texts)))
        tokens = [PAD_TOKEN, UNK_TOKEN] + vocab
        stoi = {ch: i for i, ch in enumerate(tokens)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, pad_id=stoi[PAD_TOKEN], unk_id=stoi[UNK_TOKEN])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, UNK_TOKEN) for i in ids)

    def save(self, path: str) -> None:
        write_json(path, {"stoi": self.stoi, "pad_id": self.pad_id, "unk_id": self.unk_id})

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        payload = read_json(path)
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {int(v): str(k) for k, v in stoi.items()}
        return cls(stoi=stoi, itos=itos, pad_id=int(payload["pad_id"]), unk_id=int(payload["unk_id"]))


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
