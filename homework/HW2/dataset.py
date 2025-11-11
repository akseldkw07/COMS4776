# datasets.py
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset


SPECIALS = {"SOS": "<SOS>", "EOS": "<EOS>", "EOP": "<EOP>"}


def _read_lines(file_path: Path) -> List[str]:
    return file_path.read_text(encoding="utf-8").strip().lower().splitlines()


def _read_pairs(file_path: Path) -> Tuple[List[str], List[str]]:
    src, tgt = [], []
    for line in _read_lines(file_path):
        if not line.strip():
            continue
        en, pig = line.strip().split()
        src.append(en)
        tgt.append(pig)
    return src, tgt


def _all_alpha_or_dash(s: str) -> bool:
    return all(c.isalpha() or c == "-" for c in s)


def _filter(lines: List[str]) -> List[str]:
    return [l for l in lines if _all_alpha_or_dash(l)]


def _build_vocab(src: List[str], tgt: List[str]) -> Dict[str, int]:
    charset = sorted(set("".join(src) + "".join(tgt)))
    token_to_idx = {ch: i for i, ch in enumerate(charset)}

    # reserve three extra indices for special symbols
    start_idx = len(token_to_idx)
    end_idx = start_idx + 1
    eop_idx = start_idx + 2

    token_to_idx[SPECIALS["SOS"]] = start_idx
    token_to_idx[SPECIALS["EOS"]] = end_idx
    token_to_idx[SPECIALS["EOP"]] = eop_idx
    return token_to_idx


class _CharDatasetBase(Dataset):
    """
    Common utilities shared by EncoderDataset and DecoderDataset
    """

    def __init__(self, data_fpath: str, filename: str):
        self._data_path = Path(data_fpath) / f"{filename}.txt"
        src, tgt = _read_pairs(self._data_path)
        self.src = _filter(src)
        self.tgt = _filter(tgt)

        self.token_to_idx: Dict[str, int] = _build_vocab(self.src, self.tgt)
        self.idx_to_token: Dict[int, str] = {
            idx: tok for tok, idx in self.token_to_idx.items()
        }

        self.start_idx = self.token_to_idx[SPECIALS["SOS"]]
        self.end_idx = self.token_to_idx[SPECIALS["EOS"]]
        self.eop_idx = self.token_to_idx[SPECIALS["EOP"]]

        # Store paired list for easy indexing
        self.pairs: List[Tuple[str, str]] = list(set(zip(self.src, self.tgt)))

    
    def to_idx(self, s: str) -> List[int]:
        try:
            return [self.token_to_idx[ch] for ch in s]
        except KeyError as e:
            raise KeyError(f"Character {e} not in vocabulary.") from None

    def to_token(self, idxs: List[int]) -> str:
        try:
            return "".join(self.idx_to_token[i] for i in idxs)
        except KeyError as e:
            raise KeyError(f"Index {e} not in vocabulary.") from None


class DecoderDataset(_CharDatasetBase):
    """
    __getitem__ returns (encoder_input , encoder_target)

    encoder_input  =  [SOS] + en_idx + [EOP] + pig_idx
    encoder_target =          en_idx + [EOP] + pig_idx + [EOS]
    """

    def __getitem__(self, idx):
        en, pig = self.pairs[idx]
        en_idx = self.to_idx(en)
        pig_idx = self.to_idx(pig)

        inp = (
            [self.start_idx]
            + en_idx
            + [self.eop_idx]
            + pig_idx
        )
        tgt = (
            en_idx
            + [self.eop_idx]
            + pig_idx
            + [self.end_idx]
        )
        return torch.tensor(inp, dtype=torch.long), torch.tensor(
            tgt, dtype=torch.long
        )

    def __len__(self):
        return len(self.pairs)

    # ---------  Collate -------------
    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        max_len_inp = max(len(x) for x in inputs)
        max_len_tgt = max(len(x) for x in targets)

        def pad(seq, L):
            pad_len = L - len(seq)
            if pad_len:
                return torch.cat(
                    [seq, torch.full((pad_len,), self.end_idx)]
                )
            return seq

        inputs = torch.stack([pad(s, max_len_inp) for s in inputs])
        targets = torch.stack([pad(t, max_len_tgt) for t in targets])
        return inputs, targets



class EncoderDecoderDataset(_CharDatasetBase):
    """
    __getitem__ returns  (decoder_input , annotation , decoder_target)

    decoder_input  =  [SOS] + pig_idx
    annotation     =  en_idx
    decoder_target =  pig_idx + [EOS]
    """

    def __getitem__(self, idx):
        en, pig = self.pairs[idx]
        en_idx = self.to_idx(en)
        pig_idx = self.to_idx(pig)

        dec_in = [self.start_idx] + pig_idx
        dec_tgt = pig_idx + [self.end_idx]

        return (
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(en_idx, dtype=torch.long),
            torch.tensor(dec_tgt, dtype=torch.long),
        )

    def __len__(self):
        return len(self.pairs)

    # ----------  Collate ------------
    def collate_fn(self, batch):
        dec_ins, annos, dec_tgts = zip(*batch)
        L_in = max(len(x) for x in dec_ins)
        L_anno = max(len(a) for a in annos)
        L_tgt = max(len(t) for t in dec_tgts)

        def pad(seq, L):
            pad_len = L - len(seq)
            if pad_len:
                return torch.cat(
                    [seq, torch.full((pad_len,), self.end_idx)]
                )
            return seq

        dec_ins = torch.stack([pad(s, L_in) for s in dec_ins])
        annos = torch.stack([pad(a, L_anno) for a in annos])
        dec_tgts = torch.stack([pad(t, L_tgt) for t in dec_tgts])

        return dec_ins, annos, dec_tgts
