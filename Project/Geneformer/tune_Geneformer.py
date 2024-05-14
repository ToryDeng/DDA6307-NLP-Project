import argparse, sys, datetime, glob, os, gc, json, warnings

sys.path.insert(0, "/home/dengtao/nlp-course/Geneformer")
warnings.filterwarnings("ignore")
from geneformer import Classifier


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the random seed (default: 0)", default=0)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Directory storing the dataset",
    default="../datasets/human_dcm_hcm_nf_2048_w_length.dataset",
)
parser.add_argument("--model_dir", type=str, help="Directory storing the model", required=False)
parser.add_argument(
    "--pretrained_layers", type=int, help="The number of layers of the pretrained model", default=6, choices=[6, 12]
)
parser.add_argument(
    "--epochs",
    type=float,
    help="Total number of training epochs to perform (default: 1, if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    default=1,
)
parser.add_argument("--learning_rate", type=float, help="The learning rate (default: 1e-4)", default=1e-4)
parser.add_argument("--train_batch_size", type=int, help="The batch_size in training (default: 16)", default=16)
parser.add_argument(
    "--forward_batch_size", type=int, help="The batch_size in evaluation or prediction (default: 64)", default=64
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    help="Number of steps used for a linear warmup from 0 to `learning_rate` (default: 500).",
    default=500,
)
parser.add_argument(
    "--weight_decay",
    type=float,
    help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in `AdamW` optimizer (default: 0.1)",
    default=0.1,
)
parser.add_argument(
    "--no_fine_tune", help="Whether to fine tune the pretrained model on the dataset", action="store_true"
)
parser.add_argument(
    "--tune_method",
    type=str,
    help="The fine-tuning method (default: not to use any fine-tuning method)",
    choices=["lora", "qlora"],
    required=False,
)
parser.add_argument(
    "--hyperopt_trials",
    type=int,
    help="Number of trials to run for hyperparameter optimization (default: 0)",
    default=0,
)
parser.add_argument("--workers", type=int, help="The number of CPU processes to use (default: 32)", default=32)
parser.add_argument(
    "--freeze_layers", type=int, help="The number of layers to freeze from fine-tuning (default: 0)", default=0
)

args = parser.parse_args()

# configure the number of available GPUs before importing torch
if args.hyperopt_trials == 0:  # run rays' trials in parallel when args.hyperopt_trials > 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

# get time stamps
current_datetime = datetime.datetime.now()
datetime_stamp, date_stamp = current_datetime.strftime("%y%m%d%H%M%S"), current_datetime.strftime("%y%m%d")

# prepare directories
output_prefix = "classifier_test"
# convert relative path to absolute path to resolve the error `pyarrow.lib.ArrowInvalid: URI has empty scheme`
output_dir = os.path.abspath(f"../ckpt/{datetime_stamp}")


def remove_files(pattern):
    num_removed_files = 0
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            num_removed_files += 1
        except OSError as e:
            print(f"Error: {e.strerror} - {file_path}")
    print(f"Removed: {num_removed_files} files.")


remove_files(os.path.join(args.dataset_dir, "cache*"))
remove_files(os.path.join(args.dataset_dir, "tmp*"))
os.makedirs(output_dir)

# specify data to be filtered
filter_data_file = os.path.join(args.dataset_dir, "filter_data.json")
if os.path.exists(filter_data_file):
    with open(filter_data_file, "r") as f:
        filter_data_dict = json.load(f)
        print(f"Use the settings in {filter_data_dict} to filter data.")
else:
    filter_data_dict = None

# configure the Classifier
training_args = {
    "num_train_epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    "per_device_train_batch_size": args.train_batch_size,
    "seed": args.seed,
    "run_name": f"{args.pretrained_layers}layers_{datetime_stamp}",
}
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data=filter_data_dict,
    training_args=training_args,
    max_ncells=None,
    freeze_layers=args.freeze_layers,
    num_crossval_splits=1,
    forward_batch_size=args.forward_batch_size,
    nproc=args.workers,
)

# load subjects in train, eval, and test datasets
splits_ids_file = os.path.join(args.dataset_dir, "train_eval_test_subjects.json")
if os.path.exists(splits_ids_file):
    with open(splits_ids_file, "r") as f:
        splits_ids = json.load(f)
        train_ids, eval_ids, test_ids = splits_ids["train_ids"], splits_ids["eval_ids"], splits_ids["test_ids"]
else:
    raise OSError(f"The subject ids in train, eval, and test datasets should be saved in {args.dataset_dir}.")

train_test_id_split_dict = {"attr_key": "individual", "train": train_ids + eval_ids, "test": test_ids}
cc.prepare_data(
    input_data_file=args.dataset_dir,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_test_id_split_dict,
)
train_valid_id_split_dict = {"attr_key": "individual", "train": train_ids, "eval": eval_ids}

# fine tune or directly test the pretrained model
# convert to absolute path to resolve the model not found error in `transformers`
if args.model_dir is None:
    model_dir = os.path.abspath(f"../models/Geneformer_{args.pretrained_layers}layers/")
else:
    model_dir = os.path.abspath(args.model_dir)

if args.no_fine_tune:
    print(f"Directly test the model stored at {model_dir} without any fine-tuning...")
    test_model_dir = model_dir
else:
    test_model_dir = f"{output_dir}/{date_stamp}_geneformer_cellClassifier_{output_prefix}/ksplit1/"
    # fine-tuning starts
    all_metrics = cc.validate(
        model_directory=model_dir,
        prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        output_directory=output_dir,
        output_prefix=output_prefix,
        split_id_dict=train_valid_id_split_dict,
        tune_method=args.tune_method,
        n_hyperopt_trials=args.hyperopt_trials,
    )
    # fine-tuning ends; free GPU memory
    gc.collect()
    torch.cuda.empty_cache()

# if fine tuning, test the saved (pretrained) model
if args.hyperopt_trials == 0:
    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "disease", "states": "all"},
        forward_batch_size=args.forward_batch_size,
        nproc=args.workers,
    )
    all_metrics_test = cc.evaluate_saved_model(
        model_directory=test_model_dir,
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
        tune_method=args.tune_method,
    )
    print(all_metrics_test)
