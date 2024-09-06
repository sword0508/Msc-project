import os
import sys
import math
import copy
import argparse
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from loguru import logger

logger.add("text_eeg_multimodal_model.log", level="INFO")
# 手机 和手表 （可穿戴设备）
# * 音频
# * 视频
# * PPG （心率）

# paper
# EEG Encoder
# 数据集
# fusion is ok


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")


class TextEEGDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_length=128, eeg_scaler=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eeg_scaler = eeg_scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["new_words"]
        # print(text)
        x = self.tokenizer(  # transformers,
            text,
            padding="max_length",
            max_length=self.max_seq_length,  # 256
            truncation=True,
            return_tensors="np",  # numpy.ndarray/tensor, torch.Tensor
        )

        eeg = self.eeg_scaler.transform(
            np.array(
                self.df.iloc[idx][
                    [
                        "delta0",
                        "delta1",
                        "delta2",
                        "delta3",
                        "delta4",
                        "delta5",
                        "lowAlpha0",
                        "lowAlpha1",
                        "lowAlpha2",
                        "lowAlpha3",
                        "lowAlpha4",
                        "lowAlpha5",
                        "highAlpha0",
                        "highAlpha1",
                        "highAlpha2",
                        "highAlpha3",
                        "highAlpha4",
                        "highAlpha5",
                        "lowBeta0",
                        "lowBeta1",
                        "lowBeta2",
                        "lowBeta3",
                        "lowBeta4",
                        "lowBeta5",
                        "highBeta0",
                        "highBeta1",
                        "highBeta2",
                        "highBeta3",
                        "highBeta4",
                        "highBeta5",
                        "lowGamma0",
                        "lowGamma1",
                        "lowGamma2",
                        "lowGamma3",
                        "lowGamma4",
                        "lowGamma5",
                        "middleGamma0",
                        "middleGamma1",
                        "middleGamma2",
                        "middleGamma3",
                        "middleGamma4",
                        "middleGamma5",
                        "theta0",
                        "theta1",
                        "theta2",
                        "theta3",
                        "theta4",
                        "theta5",
                    ]
                ]
            ).reshape(1, -1)
        )

        return {
            # 词序列，[235, 687, 989, ..., 111, 111, 111,]
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],  # [0,0,0,.....,1,1,1]
            # eeg序列 (freq bonds, seq_len)
            "eeg": eeg.astype(np.float32).reshape(8, 6),
            "y": self.df.iloc[idx]["label"],
        }

# black format


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class EEGConformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=64,
            out_channels=args.eeg_conformer_k,
            kernel_size=(8, 1),
            stride=1,
            padding=1,
            bias=False,
        )
        self.positional_encoding = positionalencoding1d(
            args.eeg_conformer_k, 20)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.eeg_conformer_k, nhead=args.eeg_conformer_nheads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.eeg_conformer_nlayers
        )

    def forward(self, x):
        # x: (B, bond, seq_len)
        x = x.unsqueeze(1)  # (B, 1, bond, seq_len) NCHW
        x = self.temporal_conv(x)
        x = F.gelu(x)  # add gelu activation
        x = self.spatial_conv(x)
        # print("x.shape: ", x.shape)  # torch.Size([B, eeg_conformer_k, seq_len, bonds])
        x = x.mean(-1).squeeze(-1)  # (B, eeg_conformer_k, seq_len)
        x = x.transpose(1, 2)  # (B, seq_len, eeg_conformer_k)
        x = x + \
            self.positional_encoding[: x.shape[1], : x.shape[2]].to(x.device)
        x = self.transformer_encoder(x)
        # print("x.shape: ", x.shape)  # torch.Size([B, seq_len, eeg_conformer_k])

        return x


class TextEEGModel(nn.Module):
    def __init__(self, text_encoder, eeg_encoder, args):
        super().__init__()
        self.text_encoder = text_encoder
        self.eeg_encoder = eeg_encoder
        self.fusion_strategy = args.fusion_strategy

        if self.fusion_strategy == "concat":
            self.fc = nn.Linear(
                args.text_encoder_dim + args.eeg_conformer_k, args.concat_project_dim
            )
            self.classifier = nn.Linear(args.concat_project_dim, args.ncalss)
        elif self.fusion_strategy == "transformer":
            self.text_projection = nn.Linear(
                args.text_encoder_dim, args.eeg_conformer_k
            )
            self.positional_encoding = positionalencoding1d(
                args.eeg_conformer_k, 50)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.eeg_conformer_k, nhead=args.eeg_conformer_nheads
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=args.eeg_conformer_nlayers
            )
            self.final_layer_norm = nn.LayerNorm(args.eeg_conformer_k)
            self.classifier = nn.Linear(args.eeg_conformer_k, args.ncalss)
        elif self.fusion_strategy == "biaffine":
            self.text_projection = nn.Linear(
                args.text_encoder_dim, args.eeg_conformer_k
            )
            self.classifier = nn.Linear(args.eeg_conformer_k * 2, args.ncalss)
        elif self.fusion_strategy == "bottleneck":
            self.text_projection = nn.Linear(
                args.text_encoder_dim, args.eeg_conformer_k
            )
            self.bottle_neck = nn.Parameter(
                torch.randn(1, args.bottleneck_sequence_length,
                            args.eeg_conformer_k)
            )
            self.fc = nn.Linear(args.eeg_conformer_k, args.concat_project_dim)
            self.classifier = nn.Linear(args.concat_project_dim, args.ncalss)
        else:
            raise ValueError(
                f"Invalid fusion strategy: {self.fusion_strategy}")

    def forward(self, text_input_ids, text_attention_mask, eeg):
        eeg_encoding = self.eeg_encoder(eeg)  # (B, seq_len, eeg_conformer_k)
        text_encoding = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_hidden_states=False,
        )
        text_encoding = text_encoding[
            "last_hidden_state"
        ]  # (B, seq_len, text_encoder_dim)
        # print("text_encoding.shape: ", text_encoding.shape)
        # print("eeg_encoding.shape: ", eeg_encoding.shape)
        if self.fusion_strategy == "concat":
            eeg_encoding = eeg_encoding.mean(
                1).squeeze(1)  # (B, eeg_conformer_k)
            text_encoding = text_encoding.mean(
                1).squeeze(1)  # (B, text_encoder_dim)
            x = torch.cat(
                [eeg_encoding, text_encoding], dim=-1
            )  # (B, text_encoder_dim + eeg_conformer_k)
            x = self.fc(x)
            x = F.gelu(x)
            x = self.classifier(x)
            return x

        elif self.fusion_strategy == "transformer":
            text_encoding = self.text_projection(text_encoding)
            x = torch.cat(
                [eeg_encoding, text_encoding], dim=1
            )  # (B, seq_len, eeg_conformer_k + text_encoder_dim)
            x = x + \
                self.positional_encoding[: x.shape[1],
                                         : x.shape[2]].to(x.device)
            x = self.transformer_encoder(x)
            x = self.final_layer_norm(x)
            x = x.mean(1).squeeze(1)
            x = self.classifier(x)

            return x
        elif self.fusion_strategy == "biaffine":
            text_encoding = self.text_projection(
                text_encoding
            )  # (B, L1, eeg_conformer_k)
            # eeg_encoding   # (B, L2, eeg_conformer_k, )
            left = torch.matmul(
                text_encoding, eeg_encoding.transpose(1, 2)
            )  # (B, L1, L2)
            right = torch.matmul(
                eeg_encoding, text_encoding.transpose(1, 2)
            )  # (B, L2, L1)
            left = F.softmax(left, dim=-1)  # (B, L1, L2)
            right = F.softmax(right, dim=-1)  # (B, L2, L1)
            left = torch.matmul(left, eeg_encoding)  # (B, L1, eeg_conformer_k)
            # (B, L2, eeg_conformer_k)
            right = torch.matmul(right, text_encoding)
            left = left.mean(1).squeeze(1)  # (B, eeg_conformer_k)
            right = right.mean(1).squeeze(1)
            x = torch.cat([left, right], dim=-1)  # (B, eeg_conformer_k * 2)
            x = self.classifier(F.gelu(x))

            return x

        elif self.fusion_strategy == "bottleneck":
            text_encoding = self.text_projection(text_encoding)
            concat_x = torch.cat(
                [eeg_encoding, text_encoding], dim=1
            )  # (B, L, eeg_conformer_k)
            # self.bottle_neck = self.bottle_neck.unsqueeze(
            #     0
            # )  # (1, bottleneck_sequence_length, eeg_conformer_k)
            x = torch.matmul(
                self.bottle_neck, concat_x.transpose(1, 2)
            )  # (B, bottleneck_sequence_length, L)
            x = F.softmax(x, dim=-1)  # (B, bottleneck_sequence_length, L)
            x = torch.matmul(
                x, concat_x
            )  # (B, bottleneck_sequence_length, eeg_conformer_k)
            x = x.mean(1).squeeze(1)  # (B, eeg_conformer_k)

            x = self.classifier(F.gelu(self.fc(x)))

            return x


def get_label(row):
    if row["sad_trans"] == 1:
        return 1
    elif row["happy_trans"] == 1:
        return 2
    elif row["angry_trans"] == 1:
        return 3
    elif row["nervous_trans"] == 1:
        return 4
    else:
        return 0


def evaluation(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in loader:
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            eeg = batch["eeg"].to(device)
            # print("eeg: ", eeg[0].reshape(-1))
            y = batch["y"].to(device)
            if args.model == "TextEmotionModel":
                outputs = model(input_ids, attention_mask)
            elif args.model == "EEGEmotionModel":
                outputs = model(eeg)
            else:
                outputs = model(input_ids, attention_mask, eeg)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            y_pred.extend(list(predicted.cpu().numpy()))
            y_true.extend(list(y.cpu().numpy()))
    # print(y_pred, y_true)
    correct = 0
    for a, b in zip(y_pred, y_true):
        if a == b:
            correct += 1
    # f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    model.train()

    return correct / len(y_pred), f1_weighted


class TextEmotionModel(nn.Module):
    """
    A PyTorch module for text emotion classification.

    Args:
    - text_encoder: a pre-trained text encoder model
    - args: a Namespace object containing model hyperparameters

    Attributes:
    - text_encoder: the text encoder model
    - fc: a linear layer for projecting text encoding to a concatenated projection dimension
    - classifier: a linear layer for classification

    Methods:
    - forward(text_input_ids, text_attention_mask): performs a forward pass of the model
    """

    def __init__(self, text_encoder, args) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.fc = nn.Linear(args.text_encoder_dim, args.text_encoder_dim // 2)
        self.classifier = nn.Linear(args.text_encoder_dim // 2, args.ncalss)

    def forward(self, text_input_ids, text_attention_mask):
        text_encoding = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_hidden_states=False,
        )
        text_encoding = text_encoding["last_hidden_state"]
        text_encoding = text_encoding.mean(1).squeeze(1)
        x = self.fc(text_encoding)
        x = F.gelu(x)
        x = self.classifier(x)
        return x


class EEGEmotionModel(nn.Module):
    """
    A PyTorch module for EEG-based emotion recognition.

    Args:
    - eeg_encoder (nn.Module): the EEG encoder module
    - args (argparse.Namespace): the command-line arguments

    Attributes:
    - eeg_encoder (nn.Module): the EEG encoder module
    - fc (nn.Linear): a fully connected layer
    - classifier (nn.Linear): a fully connected layer for classification
    """

    def __init__(self, eeg_encoder, args) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.fc = nn.Linear(args.eeg_conformer_k, args.eeg_conformer_k // 2)
        self.classifier = nn.Linear(args.eeg_conformer_k // 2, args.ncalss)

    def forward(self, eeg):
        """
        Forward pass of the EEGEmotionModel.

        Args:
        - eeg (torch.Tensor): the input EEG data

        Returns:
        - torch.Tensor: the output of the model
        """
        eeg_encoding = self.eeg_encoder(eeg)
        eeg_encoding = eeg_encoding.mean(1).squeeze(1)
        x = self.fc(eeg_encoding)
        x = F.gelu(x)
        x = self.classifier(x)
        return x


def main(args):
    df = pd.read_csv("df.csv")
    logger.info(f"df shape: {df.shape}")
    logger.info(f"df columns: {list(df.columns)}")
    df["label"] = (
        df["sad_trans"] + df["happy_trans"] +
        df["angry_trans"] + df["nervous_trans"]
    )
    df2 = df[(df["label"] == 0) | (df["label"] == 1)
             ].reset_index(drop=True)  # 5分类
    logger.info(f"df2 shape: {df2.shape}")
    df = df2
    df["label"] = df.apply(get_label, axis=1)
    # print(Counter(df["label"]))  # Counter({0: 521, 4: 357, 2: 184, 3: 42})
    df = df.sample(frac=1.0, random_state=args.seed, replace=False).reset_index(
        drop=True
    )
    train = df.iloc[: int(len(df) * (1 - args.val_ratio - args.test_ratio))]
    val = df.iloc[
        int(len(df) * (1 - args.val_ratio - args.test_ratio)): int(
            len(df) * (1 - args.test_ratio)
        )
    ].reset_index(drop=True)
    test = df.iloc[int(len(df) * (1 - args.test_ratio))                   :].reset_index(drop=True)
    eeg_scaler = StandardScaler()
    eeg_scaler = eeg_scaler.fit(
        train[
            [
                "delta0",
                "delta1",
                "delta2",
                "delta3",
                "delta4",
                "delta5",
                "lowAlpha0",
                "lowAlpha1",
                "lowAlpha2",
                "lowAlpha3",
                "lowAlpha4",
                "lowAlpha5",
                "highAlpha0",
                "highAlpha1",
                "highAlpha2",
                "highAlpha3",
                "highAlpha4",
                "highAlpha5",
                "lowBeta0",
                "lowBeta1",
                "lowBeta2",
                "lowBeta3",
                "lowBeta4",
                "lowBeta5",
                "highBeta0",
                "highBeta1",
                "highBeta2",
                "highBeta3",
                "highBeta4",
                "highBeta5",
                "lowGamma0",
                "lowGamma1",
                "lowGamma2",
                "lowGamma3",
                "lowGamma4",
                "lowGamma5",
                "middleGamma0",
                "middleGamma1",
                "middleGamma2",
                "middleGamma3",
                "middleGamma4",
                "middleGamma5",
                "theta0",
                "theta1",
                "theta2",
                "theta3",
                "theta4",
                "theta5",
            ]
        ].values
    )
    # print(eeg_scaler.mean_)
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_text_model_name)
    train_dataset = TextEEGDataset(
        train, text_tokenizer, args.max_text_tokens_length, eeg_scaler
    )
    val_dataset = TextEEGDataset(
        val, text_tokenizer, args.max_text_tokens_length, eeg_scaler
    )
    test_dataset = TextEEGDataset(
        test, text_tokenizer, args.max_text_tokens_length, eeg_scaler
    )
    logger.info(f"train dataset size: {len(train_dataset)}")
    logger.info(f"val dataset size: {len(val_dataset)}")
    logger.info(f"test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == "TextEmotionModel":
        text_encoder = AutoModel.from_pretrained(
            args.pretrained_text_model_name)
        model = TextEmotionModel(text_encoder, args).to(device)
    elif args.model == "EEGEmotionModel":
        eeg_encoder = EEGConformer(args)
        model = EEGEmotionModel(eeg_encoder, args).to(device)
    else:
        text_encoder = AutoModel.from_pretrained(
            args.pretrained_text_model_name)
        eeg_encoder = EEGConformer(args)
        model = TextEEGModel(text_encoder, eeg_encoder, args).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.model == "TextEmotionModel":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.model == "EEGEmotionModel":
        optimizer = optim.Adam(
            model.parameters(), lr=args.eeg_conformer_learning_rate)
    else:
        # for p, n in model.named_parameters():
        #     print("p: ", p, n.shape)
        optimizer = optim.Adam([n for p, n in model.named_parameters(
        ) if not p.startswith("eeg_encoder")], lr=args.learning_rate)
        optimizer_eeg = optim.Adam([n for p, n in model.named_parameters(
        ) if p.startswith("eeg_encoder")], lr=args.eeg_conformer_learning_rate)

    total_training_loss = 0.0
    total_training_steps = 0
    total_training_y = []
    total_training_y_hat = []
    best_model_states = None
    best_eval_acc = 0.0
    for epoch in range(args.epochs):
        for idx, batch in enumerate(train_dataloader):
            # print(batch["input_ids"].shape)  # torch.Size([B, 1, args.max_text_tokens_length])
            # print(batch["attention_mask"].shape)  # # torch.Size([B, 1, args.max_text_tokens_length])
            # print(batch["eeg"].shape)  # torch.Size([B, 8, 6]) (freq bonds, seq_len)
            # print(batch["y"].shape)
            # print("=====================================")
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            eeg = batch["eeg"].to(device)
            y = batch["y"].to(device)
            # print("y: ", y)
            # eeg_encoder(eeg)
            if args.model == "TextEmotionModel":
                y_hat = model(input_ids, attention_mask)
            elif args.model == "EEGEmotionModel":
                y_hat = model(eeg)
            else:
                y_hat = model(input_ids, attention_mask, eeg)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            if args.model == "TextEEGModel":
                optimizer_eeg.zero_grad()
            loss.backward()
            optimizer.step()
            if args.model == "TextEEGModel":
                optimizer_eeg.step()
            # print(loss.item())
            total_training_loss += loss.detach().item()
            total_training_steps += 1
            total_training_y.extend(list(y.cpu().numpy()))
            total_training_y_hat.extend(list(y_hat.argmax(-1).cpu().numpy()))
            if total_training_steps % args.evaluation_step == 0:
                # logger.info(f"total_training_steps: {total_training_steps}, total_training_loss: {total_training_loss / total_training_steps}")
                cur_training_acc = sum(
                    [
                        1.0 if a == b else 0
                        for a, b in zip(total_training_y, total_training_y_hat)
                    ]
                ) / len(total_training_y)
                # cur_training_f1_macro = f1_score(total_training_y, total_training_y_hat, average="macro")
                cur_training_f1_weighted = f1_score(
                    total_training_y, total_training_y_hat, average="weighted"
                )

                eval_acc, f1_weighted = evaluation(
                    model, val_dataloader, device)

                logger.info(
                    "Epoch: {}, Steps: {}, Train Acc: {:.2f}%, Train F1_weighted: {:.2f}%, Eval Acc: {:.2f}%, Eval F1_weighted: {:.2f}%".format(
                        epoch,
                        idx,
                        cur_training_acc * 100,
                        cur_training_f1_weighted * 100,
                        eval_acc * 100,
                        f1_weighted * 100,
                    )
                )

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_model_states = copy.deepcopy(model.state_dict())
                    torch.save(best_model_states, "best_model_states.pth")
    model.load_state_dict(best_model_states)
    test_acc, test_f1_weighted = evaluation(model, test_dataloader, device)
    logger.info(
        "Test Acc: {:.2f}, Test F1_weighted: {:.2f}".format(
            test_acc * 100, test_f1_weighted * 100
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=222,
                        help="random seed")  # 222(61.26)

    # hyper-parameters for dataset
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="validation set ratio"
    )
    parser.add_argument("--test_ratio", type=float,
                        default=0.1, help="test set ratio")

    # hyper-parameters for training
    parser.add_argument(
        "--model",
        type=str,
        default="TextEEGModel",
        choices=["TextEmotionModel", "EEGEmotionModel", "TextEEGModel"],
        help="model name",
    )
    parser.add_argument("--ncalss", type=int, default=5,
                        help="number of classes")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="training epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.00001, help="learning rate"
    )
    parser.add_argument(
        "--eeg_conformer_learning_rate", type=float, default=0.001, help="learning rate"
    )

    # hyper-parameters for model
    parser.add_argument(
        "--eeg_conformer_k", type=int, default=8, help="k for eeg conformer"
    )
    parser.add_argument(
        "--eeg_conformer_nheads", type=int, default=4, help="nheads for eeg conformer"
    )
    parser.add_argument(
        "--eeg_conformer_nlayers",
        type=int,
        default=1,
        help="num layers for eeg conformer",
    )

    parser.add_argument(
        "--pretrained_text_model_name",
        type=str,
        default="bert-base-uncased",
        help="pretrained text model name",
    )
    parser.add_argument(
        "--max_text_tokens_length",
        type=int,
        default=32,
        help="max text tokens length for bert",
    )
    parser.add_argument(
        "--text_encoder_dim", type=int, default=768, help="text encoder output dim"
    )
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default="concat",
        choices=["concat", "transformer", "biaffine", "bottleneck"],
        help="multi-modal fusion strategy",
    )
    parser.add_argument(
        "--concat_project_dim", type=int, default=128, help="concat project dim"
    )
    parser.add_argument(
        "--bottleneck_sequence_length",
        type=int,
        default=4,  # 4, 8, 12
        help="bottleneck sequence length",
    )

    # hyper-parameters for human
    parser.add_argument(
        "--evaluation_step", type=int, default=20, help="每隔多少个step在验证集上计算loss"
    )

    args = parser.parse_args()
    set_seed(args.seed)
    logger.info(args)
    main(args)
