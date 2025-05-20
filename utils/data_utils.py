import torch
import os
import pickle
import numpy as np
from config import *


class SequentialTextDataset(torch.utils.data.Dataset):
    """Dataset for sequential sampling"""

    def __init__(self, text_data, seq_len):
        self.text_data = text_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text_data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.text_data[idx : idx + self.seq_len]
        y = self.text_data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class RandomTextDataset(torch.utils.data.Dataset):
    """Dataset that returns random sequences"""

    def __init__(self, text_data, seq_len):
        self.text_data = text_data
        self.seq_len = seq_len
        self.max_idx = len(text_data) - seq_len - 1
        if self.max_idx < 0:
            raise ValueError(f"Text too small ({len(text_data)}) for seq_len {seq_len}")

    def __len__(self):
        return self.max_idx

    def __getitem__(self, _):
        # Pick random start
        start = np.random.randint(0, self.max_idx + 1)
        x = self.text_data[start : start + self.seq_len]
        y = self.text_data[start + 1 : start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_separate_datasets(
    train_path, val_path, test_path, seq_len, batch_size, save_path=None
):
    """Load and process pre-split datasets"""
    # Load files
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()

    with open(test_path, "r", encoding="utf-8") as f:
        test_text = f.read()

    # Create vocab from all data
    full_text = train_text + val_text + test_text
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)

    # Create char mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert to indices
    train_indices = [char_to_idx[ch] for ch in train_text]
    val_indices = [char_to_idx[ch] for ch in val_text]
    test_indices = [char_to_idx[ch] for ch in test_text]

    print(
        f"Data sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )

    # Save processed data
    if save_path:
        processed_data = {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": vocab_size,
        }
        with open(save_path, "wb") as f:
            pickle.dump(processed_data, f)

    # Create datasets
    train_ds = RandomTextDataset(train_indices, seq_len)
    val_ds = SequentialTextDataset(val_indices, seq_len)
    test_ds = SequentialTextDataset(test_indices, seq_len)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return (train_loader, val_loader, test_loader), char_to_idx, idx_to_char, vocab_size


def load_and_process_data(batch_size=BATCH_SIZE):
    """Main data loading function"""
    return load_separate_datasets(
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
        TEST_DATA_PATH,
        SEQ_LENGTH,
        batch_size,
        PROCESSED_DATA_PATH,
    )


def get_raw_text():
    """Get raw text for generation and testing"""
    try:
        with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
            val_text = f.read()
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_text = f.read()
        return train_text + val_text + test_text
    except:
        # Fallback to processed data if raw files not available
        if os.path.exists(PROCESSED_DATA_PATH):
            with open(PROCESSED_DATA_PATH, "rb") as f:
                data = pickle.load(f)

            # Rebuild text from indices
            train_indices = data.get("train_indices", [])
            val_indices = data.get("val_indices", [])
            test_indices = data.get("test_indices", [])

            if "idx_to_char" in data:
                text = ""
                for idx in train_indices + val_indices + test_indices:
                    text += data["idx_to_char"][idx]
                return text
        print("Couldn't get raw text!")
        return None
