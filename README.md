# Course Project of DDA6307: Natural Language Processing
This repo contains all code used in my course project of [DDA6307: Natural Language Processing](https://www.cuhk.edu.cn/en/course/11626).

## Program structure
- `Geneformer/`: The modified source code of [`Geneformer`](https://huggingface.co/ctheodoris/Geneformer). The main change is the implementation of [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) in cell classification, using [PEFT](https://huggingface.co/docs/peft/en/index) and [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index).
- `Project/`: Code to run `Geneformer` and [`scGPT`](https://github.com/bowang-lab/scGPT), streamlined based on their tutorial Jupyter Notebooks ([`Geneformer`](https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/cell_classification.ipynb), [`scGPT`](https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html)).

## Datasets
Currently, only the example datasets used by `Geneformer` and `scGPT` are supported and tested.

- `Geneformer`: [Heart disease classification](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset).
- `scGPT`: [Cell-type classification](https://drive.google.com/drive/folders/1Qd42YNabzyr2pWt9xoY4cVMTAxsNBt4v?usp=sharing).

You can make a new folder `datasets` in `Project/` where you can store the datasets. The heart disease dataset can be stored at `datasets/human_dcm_hcm_nf_2048_w_length.dataset/`, and the cell-type dataset can be stored at `Project/datasets/ms/`.


## Models
- `Geneformer`: [6-layer](https://huggingface.co/ctheodoris/Geneformer/tree/main) and [12-layer](https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer-12L-30M) models.
- `scGPT`: The [model](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) pretrained on whole-human cells.

Similar to datasets, to ensure you can successfully run the code, make a new folder `Project/models` and store the above models in these folders:
- `Geneformer_6layers/`
- `Geneformer_12layers/`
- `scGPT_human/`


## Requirements
First of all, you should create a conda environment from the yml file and activate it:

```bash
conda env create -f environment.yml
conda activate glm
```
Then you need to install [`wandb`](https://pypi.org/project/wandb/) and log into W&B first. See the [quick start](https://docs.wandb.ai/quickstart) guide. You also need to modify [one line of code](./Project/Geneformer/tune_Geneformer.py#L4) to add the `Geneformer/` path to system paths.


## Reproducibility experiments
To reproduce all results of Geneformer in the report:

```bash
cd ./Project/Geneformer/
bash run.sh
```

Although the report doesn'y include, the original results shown in the [scGPT tutorial](https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html) can be reproduced by

```bash
cd ./Project/scGPT/
python tune_scGPT.py --train --fast_transformer --amp
```

## References
> Theodoris, Christina V., et al. "Transfer learning enables predictions in network biology." Nature 618.7965 (2023): 616-624.
>
> Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods (2024): 1-11.