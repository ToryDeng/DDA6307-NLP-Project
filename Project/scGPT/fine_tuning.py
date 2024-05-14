import shutil, json, time, warnings, copy
import torch
import numpy as np
import scanpy as sc

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from typing import Tuple, Union
from argparse import Namespace
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel

from data import tokenize_train_valid, tokenize_test, prepare_dataloader

warnings.filterwarnings("ignore")


def load_vocab(args: Namespace) -> GeneVocab:
    vocab = GeneVocab.from_file(args.vocab_file)
    shutil.copy(args.vocab_file, args.save_dir / "vocab.json")
    for s in args.special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    return vocab


def load_pretrained_model(args: Namespace) -> nn.Module:
    with open(args.model_config_file, "r") as f:
        model_configs = json.load(f)
    args.logger.info(
        f"Resume model from {args.model_file}, the model args will override the " f"config {args.model_config_file}."
    )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        len(args.vocab),  # size of vocabulary
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        n_cls=args.le.classes_.shape[0],
        vocab=args.vocab,
        dropout=model_configs["dropout"],
        pad_token=args.pad_token,
        pad_value=args.pad_value,
        n_input_bins=args.n_bins,
        ecs_threshold=0.0,
        use_fast_transformer=args.fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=args.pre_norm,
    )
    try:
        model.load_state_dict(torch.load(args.model_file))
        args.logger.info(f"Loading all model params from {args.model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_file)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(args.device)
    param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    print(f"The model has {param_count} trainable parameters.")
    return model


def train(args: Namespace, model: nn.Module, loader: DataLoader) -> None:
    """Train the model for one epoch."""
    model.train()
    total_loss, total_cls, total_error = 0.0, 0.0, 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(args.device)
        input_values = batch_data["values"].to(args.device)
        celltype_labels = batch_data["celltype_labels"].to(args.device)

        src_key_padding_mask = input_gene_ids.eq(args.vocab[args.pad_token])
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask, CLS=True)

            loss = 0.0
            metrics_to_log = {}

            loss_cls = args.criterion_cls(output_dict["cls_output"], celltype_labels)
            loss = loss + loss_cls
            metrics_to_log.update({"train/cls": loss_cls.item()})

            error_rate = 1 - (
                (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item()
            ) / celltype_labels.size(0)

        model.zero_grad()
        args.scaler.scale(loss).backward()
        args.scaler.unscale_(args.optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False if args.scaler.is_enabled() else True
            )
            if len(w) > 0:
                args.logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {args.scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        args.scaler.step(args.optimizer)
        args.scaler.update()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_error += error_rate
        if batch % args.log_interval == 0 and batch > 0:
            args.lr = args.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_cls = total_cls / args.log_interval
            cur_error = total_error / args.log_interval

            args.logger.info(
                f"| epoch {args.cur_epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {args.lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | cls {cur_cls:5.2f} | err {cur_error:5.2f} | "
            )
            total_loss, total_cls, total_error = 0, 0, 0
            start_time = time.time()


def evaluate(
    args: Namespace, model: nn.Module, loader: DataLoader, return_raw: bool = False
) -> Union[Tuple[float, float], np.ndarray]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(args.device)
            input_values = batch_data["values"].to(args.device)
            celltype_labels = batch_data["celltype_labels"].to(args.device)

            src_key_padding_mask = input_gene_ids.eq(args.vocab[args.pad_token])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask, CLS=True)
                output_values = output_dict["cls_output"]
                loss = args.criterion_cls(output_values, celltype_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


def test(args: Namespace, model: nn.Module, adata_test: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    tokenized_test, celltypes_labels = tokenize_test(args, adata_test)
    test_loader = prepare_dataloader(args, tokenized_test, celltypes_labels, batch_size=args.batch_size)

    model.eval()
    predictions = evaluate(args, model=model, loader=test_loader, return_raw=True)

    # compute accuracy, precision, recall, f1
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    args.logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, " f"Macro F1: {macro_f1:.3f}"
    )

    return predictions, celltypes_labels


def fine_tune(args: Namespace, model: nn.Module, adata: sc.AnnData) -> nn.Module:
    tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels = tokenize_train_valid(args, adata)
    args.criterion_cls = nn.CrossEntropyLoss()
    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8)
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, args.schedule_interval, gamma=args.schedule_ratio)
    args.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, args.epochs + 1):
        args.cur_epoch = epoch
        epoch_start_time = time.time()
        train_loader = prepare_dataloader(
            args, tokenized_train, train_celltype_labels, batch_size=args.batch_size, verbose_in_train=True
        )
        valid_loader = prepare_dataloader(args, tokenized_valid, valid_celltype_labels, batch_size=args.batch_size)

        if args.do_train:
            train(args, model, loader=train_loader)
        val_loss, val_err = evaluate(args, model=model, loader=valid_loader)
        elapsed = time.time() - epoch_start_time
        args.logger.info("-" * 89)
        args.logger.info(
            f"| end of epoch {args.cur_epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        args.logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            args.logger.info(f"Best model with score {best_val_loss:5.4f}")

        args.scheduler.step()

    return best_model
