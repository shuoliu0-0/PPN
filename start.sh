#!/bin/bash
export PYTHONPATH=`pwd`
export PYTHON=/root/anaconda3/bin/python
export PATH=/root/anaconda3/bin:$PATH
PIP=/root/anaconda3/bin/pip
export LD_LIBRARY_PATH=/root/anaconda3/lib:$LD_LIBRARY_PATH
$PIP install --upgrade pip
pip install pandas_flavor
$PYTHON setup.py install

#$PIP install chemprop
#$PIP install scikit_learn==0.21
#$PIP install torchtext==0.4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,
# cat /usr/local/cuda/version.txt

python3 gpu_turbo.py &
pid=$!

nohup python3 gpu_turbo.py &
nohup python3 gpu_turbo.py &

# pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

#提取分子特征
# python3.6 scripts/save_features.py --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR.csv \
#                                   --save_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR.npz \
#                                   --features_generator rdkit_2d_normalized \
#                                   --restart

# python3.6 scripts/save_features.py --data_path /apdcephfs/private_sholiu/grover/lit_pcba/ESR1_ago_test.csv \
#                                   --save_path /apdcephfs/private_sholiu/grover/lit_pcba/ESR1_ago_test.npz \
#                                   --features_generator rdkit_2d_normalized \
#                                   --restart

# python3.6 scripts/save_features.py --data_path /apdcephfs/private_sholiu/grover/lit_pcba/IDH1_test.csv \
#                                   --save_path /apdcephfs/private_sholiu/grover/lit_pcba/IDH1_test.npz \
#                                   --features_generator rdkit_2d_normalized \
#                                   --restart

# python3.6 scripts/save_features.py --data_path /apdcephfs/private_sholiu/grover/lit_pcba/ADRB2_test.csv \
#                                   --save_path /apdcephfs/private_sholiu/grover/lit_pcba/ADRB2_test.npz \
#                                   --features_generator rdkit_2d_normalized \
#                                   --restart

# python3.6 scripts/save_features.py --data_path /apdcephfs/private_sholiu/grover/lit_pcba/MAPK1_test.csv \
#                                   --save_path /apdcephfs/private_sholiu/grover/lit_pcba/MAPK1_test.npz \
#                                   --features_generator rdkit_2d_normalized \
#                                   --restart
#提取原子&键字典
# python3.6 scripts/build_vocab.py --data_path /apdcephfs/private_sholiu/grover/all_smi_filtered1.csv  \
#                                 --vocab_save_folder /apdcephfs/private_sholiu/grover/smi_1/  \
#                                 --dataset_name all_smi_filtered1

#分割数据
# python3.6 scripts/split_data.py

#预训练
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.6 main.py pretrain \
#               --enable_multi_gpu \
#               --data_path /apdcephfs/private_sholiu/grover/all_smi_filtered_ss \
#               --save_dir /apdcephfs/private_sholiu/result/grover/model \
#               --atom_vocab_path /apdcephfs/private_sholiu/grover/smi_1/all_smi_filtered1_atom_vocab.pkl \
#               --bond_vocab_path /apdcephfs/private_sholiu/grover/smi_1/all_smi_filtered1_bond_vocab.pkl \
#               --feature_vocab_path /apdcephfs/private_sholiu/grover/smi_1/all_smi_filtered1_feature_vocab.pkl \
#               --graph_vocab_path /apdcephfs/private_sholiu/grover/smi_1/all_smi_filtered1_graph_vocab.pkl \
#               --batch_size 512 \
#               --dropout 0.1 \
#               --depth 5 \
#               --num_attn_head 1 \
#               --hidden_size 100 \
#               --epochs 600 \
#               --init_lr 0.00005 \
#               --max_lr 0.00008 \
#               --final_lr 0.000025 \
#               --weight_decay 0.0000001 \
#               --activation PReLU \
#               --backbone gtrans \
#               --embedding_output_type both
              
# #微调
python3.6 main.py finetune --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_train.csv \
                        --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_train.npz \
                        --save_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train \
                        --dataset_type classification \
                        --split_type scaffold_balanced \
                        --metric screening_metrics \
                        --ensemble_size 1 \
                        --num_folds 5 \
                        --no_features_scaling \
                        --ffn_hidden_size 200 \
                        --batch_size 32 \
                        --epochs 50 \
                        --init_lr 0.00015 \
                        --checkpoint_path /apdcephfs/private_sholiu/grover/smi_1/model/model.ep1000  

# #预测
python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.csv \
              --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.npz \
              --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train/fold_2 \
              --no_features_scaling \
              --output /apdcephfs/private_sholiu/grover/sholiu/GCGR_test2.csv

python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.csv \
              --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.npz \
              --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train/fold_3 \
              --no_features_scaling \
              --output /apdcephfs/private_sholiu/grover/sholiu/GCGR_test3.csv

python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.csv \
              --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.npz \
              --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train/fold_1 \
              --no_features_scaling \
              --output /apdcephfs/private_sholiu/grover/sholiu/GCGR_test1.csv

python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.csv \
              --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.npz \
              --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train/fold_0 \
              --no_features_scaling \
              --output /apdcephfs/private_sholiu/grover/sholiu/GCGR_test0.csv

python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.csv \
              --features_path /apdcephfs/private_sholiu/grover/lit_pcba/GCGR_test.npz \
              --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/GCGR_train/fold_4 \
              --no_features_scaling \
              --output /apdcephfs/private_sholiu/grover/sholiu/GCGR_test4.csv

# python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.csv \
#               --features_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.npz \
#               --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/OPRK1_nosamp_train/fold_4 \
#               --no_features_scaling \
#               --output /apdcephfs/private_sholiu/grover/sholiu/no_sample/OPRK1_test4.csv

# python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.csv \
#               --features_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.npz \
#               --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/OPRK1_nosamp_train/fold_1 \
#               --no_features_scaling \
#               --output /apdcephfs/private_sholiu/grover/sholiu/no_sample/OPRK1_test1.csv


# python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.csv \
#               --features_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.npz \
#               --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/OPRK1_nosamp_train/fold_3 \
#               --no_features_scaling \
#               --output /apdcephfs/private_sholiu/grover/sholiu/no_sample/OPRK1_test3.csv

# python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.csv \
#               --features_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.npz \
#               --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/OPRK1_nosamp_train/fold_2 \
#               --no_features_scaling \
#               --output /apdcephfs/private_sholiu/grover/sholiu/no_sample/OPRK1_test2.csv

# python main.py predict --data_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.csv \
#               --features_path /apdcephfs/private_sholiu/grover/lit_pcba/OPRK1_test.npz \
#               --checkpoint_dir /apdcephfs/private_sholiu/grover/finetune/OPRK1_nosamp_train/fold_0 \
#               --no_features_scaling \
#               --output /apdcephfs/private_sholiu/grover/sholiu/no_sample/OPRK1_test0.csv

kill $pid