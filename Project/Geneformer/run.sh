#!/bin/bash


# command to kill wandb processes: ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9

# the example dataset used by Geneformer
# directly test the pretrained model
python tune_Geneformer.py --pretrained_layers 6 --no_fine_tune
python tune_Geneformer.py --pretrained_layers 12 --no_fine_tune --forward_batch_size 32

# fine tune using the optimal hyperparameters and all parameters
python tune_Geneformer.py --pretrained_layers 6 --learning_rate 0.000804 --warmup_steps 1812 --weight_decay 0.258828 --seed 73

# fine tune using lora and the same hyperparameters
python tune_Geneformer.py --pretrained_layers 6 --learning_rate 0.000804 --warmup_steps 1812 --weight_decay 0.258828 --seed 73 --tune_method "lora"

# fine tune using qlora and the same hyperparameters
python tune_Geneformer.py --pretrained_layers 6 --learning_rate 0.000804 --warmup_steps 1812 --weight_decay 0.258828 --seed 73 --tune_method "qlora"

# fine tune using qlora and the same hyperparameters, 12 layers
python tune_Geneformer.py --pretrained_layers 12 --learning_rate 0.000804 --warmup_steps 1812 --weight_decay 0.258828 --seed 73 --tune_method "qlora" --forward_batch_size 32

# optimize hyperparameters with ray
python tune_Geneformer.py --pretrained_layers 6 --tune_method "qlora" --hyperopt_trials 30 --train_batch_size 64 --epochs 0.4
 
# fine tune using qlora and the optimized hyperparameters
python tune_Geneformer.py --pretrained_layers 6 --learning_rate 0.000953069 --warmup_steps 1083 --weight_decay 0.161494 --seed 62 --tune_method "qlora"