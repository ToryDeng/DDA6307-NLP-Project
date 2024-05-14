import argparse, time
import torch
import scgpt as scg

from pathlib import Path
from scgpt.utils import set_seed
from fine_tuning import load_vocab, load_pretrained_model, fine_tune, test
from data import load_dataset


# command to fine tune scGPT: python tune_scGPT.py --amp --fast_transformer --train
# ## Step1: Specify hyper-parameter setup for cell-type annotation task
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the random seed (default: 0)", default=0)
parser.add_argument("--dataset_name", type=str, help="the name of the dataset to use", default="ms", choices=["ms"])
parser.add_argument("--train", help="whether to train the model", action="store_true", dest="do_train")
parser.add_argument(
    "--load_model", type=str, help="folder storing the pretrained model", default="../models/scGPT_human"
)
parser.add_argument("--mask_ratio", type=float, help="the ratio of genes to mask", default=0.0)
parser.add_argument("--epochs", type=int, help="the number of training epochs", default=10)
parser.add_argument(
    "--n_bins",
    type=int,
    help="the number of consecutive intervals to divide gene expression values",
    default=51,
)
parser.add_argument("--lr", type=float, help="the learning rate", default=1e-4)
parser.add_argument("--batch_size", type=int, help="the batch_size", default=32)
parser.add_argument("--schedule_ratio", type=float, help="the ratio of epochs for learning rate schedule", default=0.9)
parser.add_argument("--fast_transformer", help="enable the fast transformer", action="store_true")
parser.add_argument("--pre_norm", help="whether to perform LayerNorm before self-attention", action="store_true")
parser.add_argument("--amp", help="whether to enable automatic mixed precision", action="store_true")

args = parser.parse_args()
print(args)

set_seed(args.seed)

# settings for input and preprocessing
args.pad_token = "<pad>"
args.special_tokens = [args.pad_token, "<cls>", "<eoc>"]
args.max_seq_len = 3001
args.schedule_interval = 1
args.log_interval = 100  # iterations
args.mask_value = -1
args.pad_value = -2

args.save_dir = Path(f"../ckpt/dev_{args.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
args.save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {args.save_dir}")
args.logger = scg.logger
scg.utils.add_file_handler(args.logger, args.save_dir / "run.log")


# Step 2: Load and pre-process data
model_dir = Path(args.load_model)
args.model_config_file = model_dir / "args.json"
args.model_file = model_dir / "best_model.pt"
args.vocab_file = model_dir / "vocab.json"

args.vocab = load_vocab(args)
adata_control, adata_test = load_dataset(args)

# Step 3: Load the pre-trained scGPT model
model = load_pretrained_model(args)

# Step 4: Finetune scGPT with task-specific objectives
best_model = fine_tune(args, model, adata_control)

# Step 5: Inference with fine-tuned scGPT model
predictions, labels = test(args, best_model, adata_test)
adata_test.obs["predictions"] = args.le.inverse_transform(predictions)

# save the model into the save_dir
torch.save(best_model.state_dict(), args.save_dir / "model.pt")
