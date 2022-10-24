"""
The cross validation function for finetuning.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/train/cross_validate.py
"""
import os
import time
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np

from grover.util.utils import get_task_names
from grover.util.utils import makedirs
from task.run_evaluation import run_evaluation
from task.train import run_training


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    k-fold cross validation.

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training with different random seeds for each fold
    all_scores = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        if args.parser_name == "finetune":
            # args.num_mt_block = 1
            # args.num_attn_head = 1
            # args.hidden_size = 100
            # args.bias = False
            # args.depth = 5
            # args.undirected = False
            # args.dense = False
            model_scores = run_training(args, time_start, logger)
        else:
            model_scores = run_evaluation(args, logger)
        all_scores.append(model_scores)
    #all_scores = np.array(all_scores)

    # Report scores for each fold
    info(f'{args.num_folds}-fold cross validation')

    for fold_num, scores in enumerate(all_scores):
        if args.metric == 'screening_metrics' and isinstance(scores[0], dict):
            info(f'Seed {init_seed + fold_num} ==> test auc = {np.nanmean(scores[0].get("auc")):.6f} ef = {np.nanmean(scores[0].get("ef")):.6f} f1={np.nanmean(scores[0].get("f1")):.6f}')
        else:
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                if args.metric == 'screening_metrics' and isinstance(score, dict):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} auc = {score.get("auc"):.6f}  ef = {score.get("ef"):.6f} f1= {score.get("f1"):.6f}')
                else:
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    re_auc, re_ef, re_f1 = [], [], []
    if args.metric == 'screening_metrics':
        for all_score in all_scores:
            if isinstance(all_score[0], dict):
                re_auc.append([all_score[0].get("auc")])
                re_ef.append([all_score[0].get("ef")])
                re_f1.append([all_score[0].get("f1")])
    if len(re_auc)!=0:
        avg_scores_auc = np.nanmean(re_auc, axis=1)
        avg_scores_ef = np.nanmean(re_ef, axis=1)
        avg_scores_f1 = np.nanmean(re_f1, axis=1)
        mean_score, std_score = np.nanmean(avg_scores_auc), np.nanstd(avg_scores_auc)
        mean_score_ef, std_score_ef = np.nanmean(avg_scores_ef), np.nanstd(avg_scores_ef)
        mean_score_f1, std_score_f1 = np.nanmean(avg_scores_f1), np.nanstd(avg_scores_f1)
        info(f'overall_{args.split_type}_test_auc={mean_score:.6f}  overall_{args.split_type}_test_ef={mean_score_ef:.6f} overall_{args.split_type}_test_f1={mean_score_f1:.6f}')
        info(f'std_auc={std_score:.6f}  std_ef={std_score_ef:.6f} std_f1={std_score_f1:.6f}')
    else:
        avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
        info(f'std={std_score:.6f}')


    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            if args.metric == 'screening_metrics' and isinstance(all_scores[0][0], dict):
                info(f'Overall test {task_name} auc = '
                     f'{np.nanmean(all_scores[:, task_num][0].get("auc")):.6f} +/- {np.nanstd(all_scores[:, task_num][0].get("auc")):.6f}')
            else:
                info(f'Overall test {task_name} {args.metric} = '
                    f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
