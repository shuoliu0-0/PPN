
A pharmacophore pre-trained method for imbalanced virtual screening

## Requirements
Our work is implementated based on PyTorch.

'''bash
python >= 3.6.8
conda >= 4.9.2
PyTorch >= 1.1.0
rdkit >= '2019.03.4'
'''

## Usage

###Pretraining

####Molecular feature extraction

'''bash
python3.6 scripts/save_features.py --data_path data/path/pretrain/data_example.csv \
                                   --save_path save/path/data_example.npz \
                                   --features_generator rdkit_2d_normalized \
                                   --restart
'''

####Build dictionaries
'''
python3.6 scripts/build_vocab.py --data_path data/path/pretrain/data_example.csv  \
                                --vocab_save_folder vocab/save/path/data_example/  \
                                --dataset_name data_example
'''

####Data Splitting
'''
python3.6 scripts/split_data.py --data_path data/path/pretrain/data_example.csv  \
                             --features_path save/path/data_example.npz  \
                             --sample_per_file 100  \
                             --output_path output/path/data_example
'''

####Pretraining
'''
python3.6 main.py pretrain \
               --enable_multi_gpu \
               --data_path output/path/data_example \
               --save_dir save/model \
               --atom_vocab_path vocab/save/path/data_example/data_example_atom_vocab.pkl \
               --bond_vocab_path vocab/save/path/data_example/data_example_bond_vocab.pkl \
               --feature_vocab_path vocab/save/path/data_example/data_example_feature_vocab.pkl \
               --graph_vocab_path vocab/save/path/data_example/data_example_graph_vocab.pkl \
               --batch_size 512 \
               --dropout 0.1 \
               --depth 5 \
               --num_attn_head 1 \
               --hidden_size 100 \
               --epochs 600 \
               --init_lr 0.00005 \
               --max_lr 0.00008 \
               --final_lr 0.000025 \
               --weight_decay 0.0000001 \
               --activation PReLU \
               --backbone gtrans \
               --embedding_output_type both
'''

###Finetuning with Existing Data
'''
python3.6 main.py finetune --data_path data/path/finetune/example.csv \
                        --features_path save/path/example.npz \
                        --save_dir save/path/finetune/example \
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
                        --checkpoint_path save/model/model.ep 
'''

###Prediction with Finetuned Model
'''
python main.py predict --data_path data/path/finetune/example_test.csv \
              --features_path save/path/example_test.npz \
              --checkpoint_dir save/path/finetune/example/fold_0 \
              --no_features_scaling \
              --output output/path/predict_example_test.csv
'''
