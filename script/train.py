import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MLP

parser = argparse.ArgumentParser()
# PATH
parser.add_argument("--save_model_path", default="../output/best_val.pt")
# train data path
parser.add_argument("--train_path", default="../data/train/train_1M_200k_50k.pt")
parser.add_argument("--train_en_path", default="../data/train/train_1M_200k_50k.pt.en")
parser.add_argument("--train_de_path", default="../data/train/train_1M_200k_50k.pt.de")
parser.add_argument("--train_zh_path", default="../data/train/train_1M_200k_50k.pt.zh")
parser.add_argument("--train_ro_path", default="../data/train/train_1M_200k_50k.pt.ro")
parser.add_argument("--train_et_path", default="../data/train/train_1M_200k_50k.pt.et")
parser.add_argument("--train_ne_path", default="../data/train/train_1M_200k_50k.pt.ne")
parser.add_argument("--train_si_path", default="../data/train/train_1M_200k_50k.pt.si")
# valid data path
parser.add_argument("--valid_path", default="../data/train/valid_1M_200k_50k.pt")
parser.add_argument("--valid_en_path", default="../data/train/valid_1M_200k_50k.pt.en")
parser.add_argument("--valid_de_path", default="../data/train/valid_1M_200k_50k.pt.de")
parser.add_argument("--valid_zh_path", default="../data/train/valid_1M_200k_50k.pt.zh")
parser.add_argument("--valid_ro_path", default="../data/train/valid_1M_200k_50k.pt.ro")
parser.add_argument("--valid_et_path", default="../data/train/valid_1M_200k_50k.pt.et")
parser.add_argument("--valid_ne_path", default="../data/train/valid_1M_200k_50k.pt.ne")
parser.add_argument("--valid_si_path", default="../data/train/valid_1M_200k_50k.pt.si")
# train parameter
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=43)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num2lang = {0: "en", 1: "de", 2: "zh", 3: "ro", 4: "et", 5: "ne", 6: "si"}


class TextDataset(Dataset):
    def __init__(self, src_emb, trg_emb, src_lang, trg_lang, src_idx, trg_idx):
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.src_idx = src_idx
        self.trg_idx = trg_idx

    def __len__(self):
        return len(self.src_emb)

    def __getitem__(self, idx):
        return {
            "src_emb": self.src_emb[idx],
            "trg_emb": self.trg_emb[idx],
            "src_lang": self.src_lang[idx],
            "trg_lang": self.trg_lang[idx],
            "src_idx": self.src_idx[idx],
            "trg_idx": self.trg_idx[idx],
        }


def rand_sentence(lang, sent_idx, data_mono):
    out, cnt = torch.unique(lang, return_counts=True)

    sort_idx = torch.argsort(lang)
    inv_idx = torch.argsort(sort_idx)

    while True:
        rand_sentence = []
        rand_emb_idx_all = []
        for lang_num, c in zip(out, cnt):
            lang = num2lang[lang_num.item()]

            emb = data_mono[lang]["emb"]
            emb_idx = data_mono[lang]["index"].to(device)
            idx = data_mono[lang + "_idx"]

            sample_idx = random.sample(idx, c)

            sample_emb = emb[sample_idx]
            sample_emb_idx = emb_idx[sample_idx]

            rand_sentence.extend(sample_emb)
            rand_emb_idx_all.extend(sample_emb_idx)

        rand_sentence = torch.stack(rand_sentence)
        rand_emb_idx_all = torch.stack(rand_emb_idx_all)

        rand_sentence = rand_sentence[inv_idx]
        rand_emb_idx_all = rand_emb_idx_all[inv_idx]

        if any(torch.eq(sent_idx, rand_emb_idx_all)):
            pass
        else:
            break

    return rand_sentence


def calculate_loss(
    src_emb,
    src_lang,
    src_me,
    src_le,
    src_li,
    trg_emb,
    trg_lang,
    trg_me,
    trg_le,
    trg_li,
    rand_src_me,
    rand_src_le,
    rand_trg_me,
    rand_trg_le,
):
    mse_fn = nn.MSELoss()
    cos_fn = nn.CosineEmbeddingLoss()
    cross_fn = nn.CrossEntropyLoss()

    y = torch.ones(src_me.size(0), device=device)
    src_lang = torch.squeeze(src_lang).long()
    trg_lang = torch.squeeze(trg_lang).long()

    loss_recon = mse_fn(src_me + src_le, src_emb) + mse_fn(trg_me + trg_le, trg_emb)

    loss_mean_close = cos_fn(src_me, trg_me, y)
    loss_mean_distant = cos_fn(src_me, rand_src_me, -y) + cos_fn(trg_me, rand_trg_me, -y)

    loss_le = cos_fn(src_le, rand_src_le, y) + cos_fn(trg_le, rand_trg_le, y)
    loss_li = cross_fn(src_li, src_lang) + cross_fn(trg_li, trg_lang)

    return loss_recon + loss_mean_distant + loss_mean_close + loss_le + loss_li


def train_model(model, dataset_train, dataset_valid, train_mono, valid_mono, optimizer, batch_size, save_model_path):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    min_valid_loss = float("inf")
    for epoch in range(1000):
        # train
        s_time = time.time()
        train_loss = 0
        for data in dataloader_train:
            src_emb = data["src_emb"].to(device)
            src_lang = data["src_lang"].to(device)
            src_idx = data["src_idx"].to(device)
            trg_emb = data["trg_emb"].to(device)
            trg_lang = data["trg_lang"].to(device)
            trg_idx = data["trg_idx"].to(device)

            # Sampling random sentence
            rand_src_emb = rand_sentence(src_lang, src_idx, train_mono).to(device)
            rand_trg_emb = rand_sentence(trg_lang, trg_idx, train_mono).to(device)

            src_me, src_le, src_li = model(src_emb)
            trg_me, trg_le, trg_li = model(trg_emb)

            rand_src_me, rand_src_le, _ = model(rand_src_emb)
            rand_trg_me, rand_trg_le, _ = model(rand_trg_emb)

            optimizer.zero_grad()
            loss = calculate_loss(
                src_emb,
                src_lang,
                src_me,
                src_le,
                src_li,
                trg_emb,
                trg_lang,
                trg_me,
                trg_le,
                trg_li,
                rand_src_me,
                rand_src_le,
                rand_trg_me,
                rand_trg_le,
            )
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # eval
        with torch.no_grad():
            valid_loss = 0
            for data in dataloader_valid:
                src_emb = data["src_emb"].to(device)
                trg_emb = data["trg_emb"].to(device)
                src_lang = data["src_lang"].to(device)
                trg_lang = data["trg_lang"].to(device)
                src_idx = data["src_idx"].to(device)
                trg_idx = data["trg_idx"].to(device)

                rand_src_emb = rand_sentence(src_lang, src_idx, valid_mono).to(device)
                rand_trg_emb = rand_sentence(trg_lang, trg_idx, valid_mono).to(device)

                src_me, src_le, src_li = model(src_emb)
                trg_me, trg_le, trg_li = model(trg_emb)

                rand_src_me, rand_src_le, _ = model(rand_src_emb)
                rand_trg_me, rand_trg_le, _ = model(rand_trg_emb)

                loss = calculate_loss(
                    src_emb,
                    src_lang,
                    src_me,
                    src_le,
                    src_li,
                    trg_emb,
                    trg_lang,
                    trg_me,
                    trg_le,
                    trg_li,
                    rand_src_me,
                    rand_src_le,
                    rand_trg_me,
                    rand_trg_le,
                )
                valid_loss += loss.item()

            print(
                f"epoch:{epoch + 1: <2}, "
                f"train_loss: {train_loss / len(dataloader_train):.5f}, "
                f"valid_loss: {valid_loss / len(dataloader_valid):.5f}, "
                f"{(time.time() - s_time) / 60:.1f}[min]"
            )

            if valid_loss < min_valid_loss:
                epochs_no_improve = 0
                min_valid_loss = valid_loss
                torch.save(model.to("cpu").state_dict(), save_model_path)
                model.to(device)
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= 10:
            break


def main():
    data_train = torch.load(args.train_path)
    dataset_train = TextDataset(
        data_train["src_emb"],
        data_train["trg_emb"],
        data_train["src_lang"],
        data_train["trg_lang"],
        data_train["src_index"],
        data_train["trg_index"],
    )

    data_valid = torch.load(args.valid_path)
    dataset_valid = TextDataset(
        data_valid["src_emb"],
        data_valid["trg_emb"],
        data_valid["src_lang"],
        data_valid["trg_lang"],
        data_valid["src_index"],
        data_valid["trg_index"],
    )

    train_mono = {
        "en": torch.load(args.train_en_path),
        "de": torch.load(args.train_de_path),
        "zh": torch.load(args.train_zh_path),
        "ro": torch.load(args.train_ro_path),
        "et": torch.load(args.train_et_path),
        "ne": torch.load(args.train_ne_path),
        "si": torch.load(args.train_si_path),
    }
    for key in list(train_mono.keys()):
        train_mono[key + "_idx"] = [i for i in range(train_mono[key]["emb"].size(0))]

    valid_mono = {
        "en": torch.load(args.valid_en_path),
        "de": torch.load(args.valid_de_path),
        "zh": torch.load(args.valid_zh_path),
        "ro": torch.load(args.valid_ro_path),
        "et": torch.load(args.valid_et_path),
        "ne": torch.load(args.valid_ne_path),
        "si": torch.load(args.valid_si_path),
    }
    for key in list(valid_mono.keys()):
        valid_mono[key + "_idx"] = [i for i in range(valid_mono[key]["emb"].size(0))]

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model, dataset_train, dataset_valid, train_mono, valid_mono, optimizer, args.batch_size, args.save_model_path
    )


if __name__ == "__main__":
    main()
