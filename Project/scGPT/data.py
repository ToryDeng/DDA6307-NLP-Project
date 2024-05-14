import scanpy as sc
import numpy as np
import torch
import os

from pathlib import Path
from argparse import Namespace
from typing import Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value


def load_dataset(args: Namespace):
    if args.dataset_name == "ms":
        data_dir = Path("../datasets/ms")
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        args.data_is_raw = False
        adata_test.var_names = adata.var["gene_name"]  # set var names
        adata = sc.concat([adata, adata_test], label="batch_id", keys=[0, 1])

    le = LabelEncoder()
    adata.obs["celltype_id"] = le.fit_transform(adata.obs["Factor Value[inferred cell type - authors labels]"])
    adata.var["id_in_vocab"] = adata.var_names.isin(args.vocab.get_itos())
    args.logger.info(
        f"match {adata.var['id_in_vocab'].sum()}/{adata.n_vars} genes in vocabulary of size {len(args.vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"]]
    args.le = le

    # set up the preprocessor, use the args to config the workflow
    # the example dataset, ms, is after QC, FS, but unnormalized and not logarithmized
    preprocessor = Preprocessor(use_key="X", log1p=args.data_is_raw, binning=args.n_bins)
    preprocessor(adata, batch_key=None)
    adata_test = adata[adata.obs["batch_id"] == 1]
    adata = adata[adata.obs["batch_id"] == 0]

    return adata, adata_test


def tokenize_train_valid(args: Namespace, adata: sc.AnnData):
    all_counts, celltypes_labels = adata.layers["X_binned"], adata.obs["celltype_id"].values.squeeze()
    train_data, valid_data, train_celltype_labels, valid_celltype_labels = train_test_split(
        all_counts, celltypes_labels, test_size=0.1, shuffle=True
    )
    gene_ids = np.array(args.vocab(adata.var_names.to_list()), dtype=int)
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=args.vocab,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=args.vocab,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
    )
    args.logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    args.logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )
    return tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels


def tokenize_test(args: Namespace, adata_test: sc.AnnData):
    gene_ids = np.array(args.vocab(adata_test.var_names.to_list()), dtype=int)
    all_counts = adata_test.layers["X_binned"]
    celltypes_labels = adata_test.obs["celltype_id"].values.squeeze()

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=args.vocab,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
    )
    return tokenized_test, celltypes_labels


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    args: Namespace,
    tokenized_data,
    celltype_labels,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    verbose_in_train: bool = False,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    masked_values_data = random_mask_value(
        tokenized_data["values"], mask_ratio=args.mask_ratio, mask_value=args.mask_value, pad_value=args.pad_value
    )
    if verbose_in_train:
        print(
            f"random masking at epoch {args.cur_epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_data == args.mask_value).sum() / (masked_values_data - args.pad_value).count_nonzero():.4f}",
        )

    tensor_celltype_labels = torch.from_numpy(celltype_labels).long()
    data_pt = {
        "gene_ids": tokenized_data["genes"],
        "values": masked_values_data,
        "celltype_labels": tensor_celltype_labels,
    }

    return DataLoader(
        dataset=SeqDataset(data_pt), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
