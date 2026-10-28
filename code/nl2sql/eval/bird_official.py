"""Pure scoring functions copied VERBATIM (logic unchanged) from the official BIRD code so that numbers are
comparable with published results:
  * soft-F1: bird-bench/mini_dev  evaluation/evaluation_f1.py @ abd11b6db92a1c9f809b32f7564c7c71b34d67f0 (MIT)
  * VES / R-VES helpers: DAMO-ConvAI bird/llm/src/evaluation_ves.py @ faf754671f5b850e1b4d1ff870db37a6a2480df6 and
    mini_dev evaluation_ves.py (reward bands)
EX itself is db.ex_match (set equality) and is cross-checked against the official execute_sql in
tests/crosscheck_bird_official.py."""
import math

import numpy as np


def calculate_row_match(predicted_row, ground_truth_row):
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage


def calculate_f1_score(predicted, ground_truth):
    if not predicted and not ground_truth:
        return 1.0
    predicted_set = set(predicted) if predicted else set()
    ground_truth_set = set(ground_truth)
    predicted = list(dict.fromkeys(predicted))
    ground_truth = list(dict.fromkeys(ground_truth))
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(pred_row, gt_row)
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)
    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)
    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_score


def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def ves_value(time_ratio):
    """official VES per-item value (percent scale applied by caller): sqrt(time_ratio)*100."""
    return math.sqrt(time_ratio) * 100


def rves_reward(time_ratio):
    """Mini-Dev R-VES reward bands."""
    if time_ratio == 0:
        return 0
    elif time_ratio >= 2:
        return 1.25
    elif time_ratio >= 1 and time_ratio < 2:
        return 1
    elif time_ratio >= 0.5 and time_ratio < 1:
        return 0.75
    elif time_ratio >= 0.25 and time_ratio < 0.5:
        return 0.5
    return 0.25
