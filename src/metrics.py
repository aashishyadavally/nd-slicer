import json
import pickle
import re
import statistics
from collections import Counter
from operator import itemgetter
from pathlib import Path

from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb

from rouge_score import rouge_scorer

from tqdm import tqdm

from utils import extract_dynamic_slice, extract_variable_flow_from_pdg

regex = '^[0-9]+$'


def compute_metrics(preds_gold_pairs):
    rouge_scores = []
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    preds = [[x for x in item['preds_topK'][0].split(' ') if re.search(regex, x[1:-1])] \
             for item in preds_gold_pairs]
    gold = [item['gold'].split(' ') for item in preds_gold_pairs]
    em_accuracy = sum([1 for i, item_preds in enumerate(preds) \
                       if item_preds == gold[i]]) / len(preds)

    rouge_scores = [rscorer.score(' '.join(item_preds),
                                  ' '.join(gold[i])) for i, item_preds in enumerate(preds)]
    results = {
        'EM-Accuracy': em_accuracy,
        'Mean ROUGE-LCS Precision': statistics.mean([score['rougeL'].precision \
                                                     for score in rouge_scores]),
        'Mean ROUGE-LCS Recall': statistics.mean([score['rougeL'].recall \
                                                  for score in rouge_scores]),
        'Mean ROUGE-LCS F1-Score': statistics.mean([score['rougeL'].fmeasure \
                                                    for score in rouge_scores]),
    }
    return results


def compute_metrics_loop(preds_gold_pairs):
    rouge_scores = []
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    preds = [[x for x in item['preds_topK'][0].split(' ') if re.search(regex, x[1:-1])] \
             for item in preds_gold_pairs]
    gold = [item['gold'].split(' ') for item in preds_gold_pairs]
    correct_cls = ['correct' if item_preds == gold[i] else 'incorrect' for i, item_preds in enumerate(preds)]
    em_accuracy = sum([1 for i, item_preds in enumerate(preds) \
                       if item_preds == gold[i]]) / len(preds)

    rouge_scores = [rscorer.score(' '.join(item_preds),
                                  ' '.join(gold[i])) for i, item_preds in enumerate(preds)]
    results = {
        'EM-Accuracy': em_accuracy,
        'Mean ROUGE-LCS Precision': statistics.mean([score['rougeL'].precision \
                                                     for score in rouge_scores]),
        'Mean ROUGE-LCS Recall': statistics.mean([score['rougeL'].recall \
                                                  for score in rouge_scores]),
        'Mean ROUGE-LCS F1-Score': statistics.mean([score['rougeL'].fmeasure \
                                                    for score in rouge_scores]),
    }
    return results, correct_cls


def compute_metrics_base(preds_gold_pairs, feats, code_examples):
    rouge_scores = []
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    trace_preds = [[int(x[1:-1]) for x in item['preds_topK'][0].split(' ') \
                    if ((re.search(regex, x[1:-1])) and (x.startswith('<')) and (x.endswith('>')))] \
                    for item in preds_gold_pairs]
    preds = []
    for i, trace in tqdm(enumerate(trace_preds)):
        input_feats = feats[i]
        variable_flow = extract_variable_flow_from_pdg(code_examples[int(input_feats.id.split('_')[0])])
        try:
            slice = extract_dynamic_slice(input_feats.criterion, trace,
                                          input_feats.occurrence, variable_flow)
        except:
            slice = []
        preds.append(slice)

    gold = [[int(x[1:-1]) for x in item['gold'].split(' ')] for item in preds_gold_pairs]
    em_accuracy = sum([1 for i, item_preds in enumerate(preds) \
                       if item_preds == gold[i]]) / len(preds)

    rouge_scores = [rscorer.score(' '.join([str(x) for x in item_preds]),
                                  ' '.join([str(x) for x in gold[i]])) for i, item_preds in enumerate(preds)]
    results = {
        'EM-Accuracy': em_accuracy,
        'Mean ROUGE-LCS Precision': statistics.mean([score['rougeL'].precision \
                                                     for score in rouge_scores]),
        'Mean ROUGE-LCS Recall': statistics.mean([score['rougeL'].recall \
                                                  for score in rouge_scores]),
        'Mean ROUGE-LCS F1-Score': statistics.mean([score['rougeL'].fmeasure \
                                                    for score in rouge_scores]),
    }
    return results


def compute_metrics_im(pairs, feats):
    preds_slices = [([int(x[1:-1]) for x in item['preds_topK'].split(' ') \
                   if re.search(regex, x[1:-1])]) for item in pairs]

    strict_correct, relaxed_correct = 0, 0
    strict_correct_preds = []
    for i, preds in enumerate(preds_slices):
        linkage = feats[i].call_linkage
        if(all(x in preds for x in [linkage['call_line'], linkage['return_line_in_callee']])):
            strict_correct += 1
            strict_correct_preds.append({'label': 'correct', 'preds': preds})
        else:
            strict_correct_preds.append({'label': 'incorrect', 'preds': preds})
        callee_lines = list(range(linkage['callee_start_line'], linkage['callee_end_line'] + 1))
        if len(set(preds).intersection(set(callee_lines))) > 0:
            relaxed_correct += 1

    strict_linkage_accuracy = strict_correct / len(preds_slices)
    relaxed_linkage_accuracy = relaxed_correct / len(preds_slices)
    results = {
        'Strict-Linkage-Accuracy': strict_linkage_accuracy,
        'Relaxed-Linkage-Accuracy': relaxed_linkage_accuracy,
    }
    return results, strict_correct_preds


def compute_metrics_crash(pairs):
    preds_slices = [([int(x[1:-1]) for x in item['preds_topK'][0].split(' ') \
                   if re.search(regex, x[1:-1])]) for item in pairs]
    reaching_statements = [set(item['reaching_statements']) for item in pairs]
    gold_slices = [item['gold'] for item in pairs]

    em_accuracy = sum([1 for i, item_preds in enumerate(preds_slices) \
                       if item_preds == gold_slices[i]]) / len(preds_slices)
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = [rscorer.score(' '.join([str(x) for x in item_preds]),
                                  ' '.join([str(x) for x in gold_slices[i]])) \
                    for i, item_preds in enumerate(preds_slices)]

    crash_accuracy, total = 0, 0
    mispredictions = []
    for i, preds in enumerate(preds_slices):
        item_reaching_statements = set(reaching_statements[i])
        item_gold = gold_slices[i]
        if len(set(item_gold).intersection(item_reaching_statements)) == 0:
            continue
        if len(set(preds).intersection(item_reaching_statements)) > 0:
            crash_accuracy += 1
        else:
            mispredictions.append([preds, item_gold])
        total += 1
    crash_accuracy /= total

    miss_rouge_scores = []
    for [miss_preds, miss_gold] in mispredictions:
        miss_rouge_scores.append(rscorer.score(' '.join([str(x) for x in miss_preds]),
                                               ' '.join([str(x) for x in miss_gold])))
    results = {
        'Crash-Accuracy': crash_accuracy,
        'EM-Accuracy': em_accuracy,
        'Mean ROUGE-LCS Precision': statistics.mean([score['rougeL'].precision \
                                                     for score in rouge_scores]),
        'Mean ROUGE-LCS Recall': statistics.mean([score['rougeL'].recall \
                                                  for score in rouge_scores]),
        'Mean ROUGE-LCS F1-Score': statistics.mean([score['rougeL'].fmeasure \
                                                    for score in rouge_scores]),
        'Misprediction Mean ROUGE-LCS F1-Score': statistics.mean([score['rougeL'].fmeasure \
                                                                  for score in miss_rouge_scores]),
    }

    return results
