import json
import pickle
import random
import re
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from utils import (
    code_to_ranges, code_to_method_calls, code_to_method_definitions, code_to_returns,
    extract_all_variables, extract_dynamic_slice, extract_dynamic_slice_with_reduced_trace,
    extract_import_lines, extract_reaching_statements, extract_trace,
    extract_variable_flow_from_pdg, get_ast_parser, string_to_tokens
)

random.seed(42)
REGEX = '^[0-9]+$'


class CrashDetectionInputFeatures(object):
    """A single training/test features for a example for extrinsic evaluation."""
    def __init__(
            self, id, code, code_tokens, slice_tokens, zero_var, criterion,
            occurrence, reaching_statements,
        ):
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.slice_tokens = slice_tokens
        self.zero_var = zero_var
        self.criterion = criterion
        self.occurrence = occurrence
        self.reaching_statements = reaching_statements


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(
            self, id, code, code_tokens, trace_tokens, slice_tokens, criterion, occurrence
        ):
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.trace_tokens = trace_tokens
        self.slice_tokens = slice_tokens
        self.criterion = criterion
        self.occurrence = occurrence


class InterMethodInputFeatures(object):
    """A single training/test features for a example for inter-method analysis."""
    def __init__(
            self, id, code_tokens, trace, criterion, occurrence, call_linkage
        ):
        self.id = id
        self.code_tokens = code_tokens
        self.trace = trace
        self.criterion = criterion
        self.occurrence = occurrence
        self.call_linkage = call_linkage


class PartialInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(
            self, original_id, id, code, code_tokens, trace_tokens, slice_tokens,
            criterion, occurrence
        ):
        self.original_id = original_id
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.trace_tokens = trace_tokens
        self.slice_tokens = slice_tokens
        self.criterion = criterion
        self.occurrence = occurrence


class PointerNetworkInputFeatures(object):
    """A single training/test features for a example in a Pointer Network Model."""
    def __init__(
            self, id, code, code_tokens, pointer_labels, slice_tokens, criterion, occurrence
        ):
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.pointer_labels = pointer_labels
        self.slice_tokens = slice_tokens
        self.criterion = criterion
        self.occurrence = occurrence


class QEInputFeatures(object):
    """A single training/test features for a example for qualitative evaluation."""
    def __init__(
            self, id, code, code_tokens, slice_tokens, criterion, occurrence, range,
        ):
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.slice_tokens = slice_tokens
        self.criterion = criterion
        self.occurrence = occurrence
        self.range = range


def convert_examples_to_features(js, tokenizer):
    return InputFeatures(
        js["id"], js["code"], js["code_tokens"], js["trace_tokens"], js["slice_tokens"],
        js["criterion"], js["occurrence"]
    )


def convert_examples_to_partial_features(js, tokenizer):
    return PartialInputFeatures(
        js["original_id"], js["id"], js["code"], js["code_tokens"], js["trace_tokens"],
        js["slice_tokens"], js["criterion"], js["occurrence"]
    )


def convert_examples_to_pointer_features(js, tokenizer):
    return PointerNetworkInputFeatures(
        js["id"], js["code"], js["code_tokens"], js["pointer_labels"], js["slice_tokens"],
        js["criterion"], js["occurrence"]
    )


def convert_zero_division_examples_to_features(js, tokenizer):
    stopping_line = js['not_zero_line']['line']
    numerator_vars = [var for var in js['not_zero_line']['vars'] if var != js['zero_var']]
    if not numerator_vars:
        numerator_var = '1'
    else:
        numerator_var = random.choice(numerator_vars)

    crash_trace = [x['line'] for x in js['reduced_trace'][:stopping_line + 1]]
    crash_trace += [crash_trace[-1] + 1]
    occurrence = crash_trace.count(stopping_line)

    criterion = stopping_line
    raw_code_lines = js['raw_code'].split('\n')
    indent_dedents_string = ' ' * (len(raw_code_lines[criterion]) - len(raw_code_lines[criterion].lstrip()))
    injected_code = '\n'.join(raw_code_lines[:criterion]) + '\n' \
                    f"{indent_dedents_string}injected_var = {numerator_var} / {js['zero_var']}\n" + \
                    '\n'.join(raw_code_lines[criterion:])
    injected_code_tokens = string_to_tokens(injected_code, tokenizer)

    reaching_statements =  extract_reaching_statements(js['zero_var'], criterion, injected_code)

    variable_flow = extract_variable_flow_from_pdg(injected_code)
    try:
        backward_slice = extract_dynamic_slice_with_reduced_trace(criterion, crash_trace, occurrence, variable_flow)
    except:
        return None

    return CrashDetectionInputFeatures(
        f"{js['id']}_{criterion}_{occurrence}", injected_code, injected_code_tokens,
        backward_slice, js['zero_var'], criterion, occurrence, reaching_statements
    )


class CodeNetTextDataset(Dataset):
    def __init__(self, tokenizer, args, mode, logger):
        self.args = args
        self.tokenizer = tokenizer
        if args.do_eval_base:
            cached_features_file = Path(args.data_dir) / f"feats_base_{mode}.pkl"
        else:
            if args.encoder == 'graphcodebert' and args.decoder == 'graphcodebert':
                cached_features_file = Path(args.data_dir) / f"feats_gcb_{mode}.pkl"
            else:
                cached_features_file = Path(args.data_dir) / f"feats_{mode}.pkl"

        filename = Path(args.data_dir) / f"{mode}-dataset.jsonl"

        if Path(cached_features_file).is_file():
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            self.examples = []
            total_num, error_num = 0, 0
            logger.info(f"Load and create features from dataset file at {filename}")
            num_lines = sum(1 for _ in open(str(filename), 'r'))
            with open(str(filename), "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines):
                    json_line = json.loads(line)
                    if len(json_line['code_tokens']) != 0:
                        trace_tokens = [f"<{trace_line['line']}>" for trace_line in json_line['trace']]
                        for criterion, criterion_slices in json_line["slices"].items():
                            for occurrence, slice in enumerate(criterion_slices):
                                if len(slice) != 1:
                                    total_num += 1
                                    code_tokens = json_line["code_tokens"]
                                    code_tokens = tokenizer.tokenize(" ".join(code_tokens))
                                    slice_tokens = [f"<{slice_line}>" for slice_line in slice]
                                    js = {
                                        "id": f"{json_line['id']}_{criterion}_{occurrence + 1}",
                                        "code": json_line["code"],
                                        "code_tokens": code_tokens,
                                        "trace_tokens": trace_tokens,
                                        "slice_tokens": slice_tokens,
                                        "criterion": int(criterion),
                                        "occurrence": occurrence + 1,
                                    }
                                    try:
                                        features = convert_examples_to_features(js, tokenizer)
                                        self.examples.append(features)
                                    except:
                                        error_num += 1

            logger.warning(f"*** Input Example Sample ***")
            for k, v in vars(self.examples[0]).items():
                print(f'{k}: {v}')
            print()

            logger.warning(f"Num examples = {len(self.examples)}")
            logger.warning(f"Error num = {error_num}")
            logger.warning(f"Saving features into cached file {cached_features_file}")

            if not Path(args.data_dir).exists():
                Path.mkdir(args.data_dir, exist_ok=True, parents=True)

            with open(str(Path(args.data_dir) / cached_features_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]

        max_source_size, max_target_size = self.args.max_source_size, self.args.max_target_size

        # Encoder-Decoder for Trace Generation
        occurrence_tokens = self.tokenizer.tokenize(str(js.occurrence))
        criterion_token = self.tokenizer.tokenize(f"<{js.criterion}>")

        source_tokens = js.code_tokens[: max_source_size - 9 - len(occurrence_tokens)]
        source_tokens = ["<s>", "<encoder-decoder>", "</s>"] + source_tokens + ["</s>"] + \
                        criterion_token + ["</s>"] + occurrence_tokens + ["<mask0>", "</s>"]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_padding_length = max_source_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id for _ in range(source_padding_length)]

        gold_tokens = js.slice_tokens[: max_target_size - 2]
        gold_tokens = ["<mask0>"] + gold_tokens + ["</s>"]
        gold_ids = self.tokenizer.convert_tokens_to_ids(gold_tokens)
        target_padding_length = max_target_size - len(gold_ids)
        gold_ids += [self.tokenizer.pad_token_id for _ in range(target_padding_length)]

        return (
               torch.tensor(source_ids),
               torch.tensor(gold_ids),
               )


class CrashDetectionDataset(Dataset):
    def __init__(self, tokenizer, args, mode, logger):
        self.args = args
        self.tokenizer = tokenizer
        cached_features_file = Path(args.data_dir) / f"feats_{mode}_crash.pkl"
        filename = Path(args.data_dir) / f'{mode}-dataset.jsonl'

        if Path(cached_features_file).is_file():
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            zero_division_examples = []
            total_num, error_num = 0, 0
            logger.info(f"Load and create features from dataset file at {filename}")
            num_lines = sum(1 for _ in open(str(filename), 'r'))
            with open(str(filename), "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines):
                    total_num += 1
                    json_line = json.loads(line)
                    trace = []
                    zero_var, where_zero_found, not_zero_line, stop = None, None, None, False
                    full_trace = [int(x['line']) for x in json_line['trace']]
                    for trace_item in json_line['trace']:
                        if not zero_var:
                            for [var, value] in trace_item['state']:
                                if value == "0":
                                    where_zero_found = int(trace_item['line'])
                                    zero_var = var

                            trace.append({
                                'line': int(trace_item['line']),
                            })
                        else:
                            for [var, value] in trace_item['state']:
                                if var == zero_var:
                                    if value != "0":
                                        stop = True
                            if stop:
                                not_zero_line = {
                                    'line': int(trace_item['line']),
                                    'vars': list([x[0] for x in trace_item['state']]),
                                }
                                break
                            else:
                                trace.append({
                                    'line': int(trace_item['line']),
                                })

                    if not zero_var: continue

                    if not not_zero_line: continue

                    if where_zero_found == not_zero_line['line']: continue

                    js = {
                        "id": f"{json_line['id']}",
                        'raw_code': json_line['code'],
                        "code_tokens": tokenizer.tokenize(" ".join(json_line['code_tokens'])),
                        'reduced_trace': trace,
                        'full_trace': full_trace,
                        'zero_var': zero_var,
                        'not_zero_line': not_zero_line,
                    }
                    try:
                        features = convert_zero_division_examples_to_features(js, tokenizer)
                        if features is not None:
                            zero_division_examples.append(features)
                    except:
                        error_num += 1

            self.examples = zero_division_examples
            logger.warning(f"*** Input Example Sample ***")
            for k, v in vars(self.examples[0]).items():
                print(f'{k}: {v}')
            print()

            logger.warning(f"Num examples = {len(self.examples)}")
            logger.warning(f"Error num = {error_num}")
            logger.warning(f"Saving features into cached file {cached_features_file}")

            if not Path(args.data_dir).exists():
                Path.mkdir(args.data_dir, exist_ok=True, parents=True)

            with open(str(Path(args.data_dir) / cached_features_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]

        max_source_size, max_target_size = self.args.max_source_size, self.args.max_target_size

        # Encoder-Decoder for Trace Generation
        occurrence_tokens = self.tokenizer.tokenize(str(js.occurrence))
        criterion_token = self.tokenizer.tokenize(f"<{js.criterion}>")

        source_tokens = js.code_tokens[: max_source_size - 9 - len(occurrence_tokens)]
        source_tokens = ["<s>", "<encoder-decoder>", "</s>"] + source_tokens + ["</s>"] + \
                        criterion_token + ["</s>"] + occurrence_tokens + ["<mask0>", "</s>"]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_padding_length = max_source_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id for _ in range(source_padding_length)]

        reaching_statements = js.reaching_statements[: max_target_size]
        padding_length = max_target_size - len(reaching_statements)
        reaching_statements += [-999 for _ in range(padding_length)]

        slice_tokens = js.slice_tokens[: max_target_size]
        padding_length = max_target_size - len(slice_tokens)
        slice_tokens += [-999 for _ in range(padding_length)]

        return (
               torch.tensor(source_ids),
               torch.tensor(reaching_statements),
               torch.tensor(slice_tokens),
               )


class InterMethodDataset(Dataset):
    def __init__(self, tokenizer, args, mode, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.ast_parser = get_ast_parser('../data-building/build')

        cached_features_file = Path(args.data_dir) / f"feats_{mode}_im.pkl"
        filename = Path('../data-building/codenetmut_test.json')

        with open(str(Path(args.data_dir) / 'train-dataset.jsonl'), "r", encoding="utf-8") as f:
            dataset_ids = [json.loads(line)['id'] for line in f]

        if Path(cached_features_file).is_file():
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            self.examples = []
            total_num = 0
            logger.info(f"Load and create features from dataset file at {filename}")
            num_lines = sum(1 for _ in open(str(filename), 'r'))
            with open(str(filename), "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines):
                    if line in dataset_ids: continue
                    try:
                        json_line = json.loads(line)
                        tree = self.ast_parser.parse(bytes(json_line['code'], 'utf8'))
                        code_lines = json_line['code'].split('\n')
                        method_definitions = code_to_method_definitions(tree.root_node)
                        method_definitions = [{'identifier': code_lines[defn['start_line']][defn['start_idx']: defn['end_idx']],
                                               'start_line': defn['start_line'],
                                               'end_line': defn['end_line']} for defn in method_definitions]
                        method_calls = code_to_method_calls(tree.root_node)
                        method_calls = [{'identifier': code_lines[call['line']][call['start_idx']: call['end_idx']],
                                         'line': call['line']} for call in method_calls]

                        identifiers = extract_all_variables(self.ast_parser, json_line['code'])
                        # Remove method definitions and method calls from list of identifiers.
                        variable_line_numbers = list(set([(item[0], item[1][0]) for item in identifiers if item[0] not in \
                                                          [defn['identifier'] for defn in method_definitions] + \
                                                          [call['identifier'] for call in method_calls]]))
                        variable_to_lines, lines_to_variable = {}, {}
                        for variable, line in variable_line_numbers:
                            if variable not in variable_to_lines:
                                variable_to_lines[variable] = [line]
                            else:
                                variable_to_lines[variable] += [line]
                            
                            if line not in lines_to_variable:
                                lines_to_variable[line] = [variable]
                            else:
                                lines_to_variable[line] += [variable]

                        method_calls = [item for item in method_calls if item['identifier'] in \
                                        [defn['identifier'] for defn in method_definitions]]
                        return_statements = [x['line'] for x in code_to_returns(tree.root_node)]

                        if len(method_definitions) <= 1: continue
                        if len(method_calls) == 0: continue

                        code_linkages = []
                        for call in method_calls:
                            for defn in method_definitions:
                                if defn['start_line'] <= call['line'] <= defn['end_line']:
                                    for _defn in method_definitions:
                                        if _defn['identifier'] == call['identifier']:
                                            callee_start_line, callee_end_line = _defn['start_line'], _defn['end_line']
                                            for return_line in return_statements:
                                                if _defn['start_line'] <= return_line <= _defn['end_line']:
                                                    return_line_in_callee = return_line

                                    code_linkages.append({'calling_start_line': defn['start_line'],
                                                          'calling_end_line': defn['end_line'],
                                                          'call_line': call['line'],
                                                          'callee_start_line': callee_start_line,
                                                          'callee_end_line': callee_end_line,
                                                          'return_line_in_callee': return_line_in_callee})

                        code_tokens = tokenizer.tokenize(" ".join(json_line['code_tokens']))
                        trace = [x['line'] for x in extract_trace(json_line['trace'])]

                        for linkage in code_linkages:
                            if linkage['call_line'] in [x[1] for x in variable_line_numbers]:
                                criterion = linkage['call_line']
                                self.examples.append(
                                    InterMethodInputFeatures(
                                        id=f"{json_line['id']}_{criterion}-1",
                                        code_tokens=code_tokens,
                                        trace=trace,
                                        criterion=criterion,
                                        occurrence=1,
                                        call_linkage=linkage,
                                    )
                                )

                            # Ignore cases where method calls don't have a variable.
                            if call['line'] not in lines_to_variable: continue
                            relevant_variables = lines_to_variable[call['line']]
                            criterion_lines = []
                            for var in relevant_variables:
                                criterion_lines += variable_to_lines[var]

                            for criterion in list(set(criterion_lines)):
                                if criterion not in trace or linkage['call_line'] not in trace: continue
                                if criterion < linkage['calling_start_line'] or criterion > linkage['calling_end_line']: continue
                                if trace.index(criterion) > trace.index(linkage['call_line']):
                                        self.examples.append(
                                            InterMethodInputFeatures(
                                                id=f"{json_line['id']}_{criterion}-1",
                                                code_tokens=code_tokens,
                                                trace=trace,
                                                criterion=criterion,
                                                occurrence=1,
                                                call_linkage=linkage,
                                            )
                                        )
                        total_num += 1
                    except Exception as e: 
                        print(e)
                        continue

            logger.warning(f"*** Input Example Sample ***")
            for k, v in vars(self.examples[0]).items():
                print(f'{k}: {v}')
            print()

            logger.warning(f"Num examples = {len(self.examples)}")

            with open(str(Path(args.data_dir) / cached_features_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]

        max_source_size, max_target_size = self.args.max_source_size, self.args.max_target_size

        # Encoder-Decoder for Trace Generation
        occurrence_tokens = self.tokenizer.tokenize(str(js.occurrence))
        criterion_token = self.tokenizer.tokenize(f"<{js.criterion}>")

        source_tokens = js.code_tokens[: max_source_size - 9 - len(occurrence_tokens)]
        source_tokens = ["<s>", "<encoder-decoder>", "</s>"] + source_tokens + ["</s>"] + \
                        criterion_token + ["</s>"] + occurrence_tokens + ["<mask0>", "</s>"]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_padding_length = max_source_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id for _ in range(source_padding_length)]

        return (
               torch.tensor(source_ids),
               )


class PartialCodeNetTextDataset(CodeNetTextDataset):
    def __init__(self, tokenizer, args, mode, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.ast_parser = get_ast_parser('../data-building/build')
        cached_features_file = Path(args.data_dir) / f"feats_{mode}_partial.pkl"

        filename = Path(args.data_dir) / f"{mode}-dataset.jsonl"

        if Path(cached_features_file).is_file():
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            self.examples = []
            total_num, error_num = 0, 0
            logger.info(f"Load and create features from dataset file at {filename}")
            num_lines = sum(1 for _ in open(str(filename), 'r'))
            with open(str(filename), "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines):
                    json_line = json.loads(line)
                    import_lines = sorted(extract_import_lines(self.ast_parser, json_line['code']))
                    if not import_lines: continue
                    import_code_lines = [json_line['code'].split('\n')[line_number] for line_number in import_lines]
                    import_code_lines = [f'{x}\n' for x in import_code_lines]
                    with open('import_lines.txt', 'a') as f:
                        f.writelines(import_code_lines)

                    importless_code = "\n".join([line for line_number, line in enumerate(json_line['code'].split('\n')) \
                                                 if line_number not in import_lines])
                    importless_code_tokens = string_to_tokens(importless_code, tokenizer)

                    line_offset_mapper = {}
                    for old_line_number in range(len(json_line['code'].split('\n'))):
                        offset = sum([1 if old_line_number > import_line_number else 0 \
                                      for import_line_number in import_lines])
                        if old_line_number not in import_lines:
                            line_offset_mapper[old_line_number] = old_line_number - offset

                    importless_trace_tokens = [f"<{line_offset_mapper[trace_line['line']]}>" \
                                               for trace_line in json_line['trace'] if trace_line['line'] in line_offset_mapper]

                    for criterion, criterion_slices in json_line['slices'].items():
                        if int(criterion) not in line_offset_mapper: continue

                        for occurrence, slice in enumerate(criterion_slices):
                                importless_slice_tokens = [f"<{line_offset_mapper[slice_line]}>" \
                                                           for slice_line in slice if slice_line in line_offset_mapper]
                                if len(importless_slice_tokens) <= 1: continue

                                importless_criterion = str(line_offset_mapper[int(criterion)])
                                js = {
                                    "original_id": f"{json_line['id']}_{criterion}_{occurrence + 1}",
                                    "id": f"{json_line['id']}_{importless_criterion}_{occurrence + 1}",
                                    "code": importless_code,
                                    "code_tokens": importless_code_tokens,
                                    "trace_tokens": importless_trace_tokens,
                                    "slice_tokens": importless_slice_tokens,
                                    "criterion": int(importless_criterion),
                                    "occurrence": occurrence + 1,
                                }
                                try:
                                    features = convert_examples_to_partial_features(js, tokenizer)
                                    total_num += 1
                                    self.examples.append(features)
                                except:
                                    error_num += 1

            logger.warning(f"*** Input Example Sample ***")
            for k, v in vars(self.examples[0]).items():
                print(f'{k}: {v}')
            print()

            logger.warning(f"Num examples = {len(self.examples)}")
            logger.warning(f"Error num = {error_num}")
            logger.warning(f"Saving features into cached file {cached_features_file}")

            if not Path(args.data_dir).exists():
                Path.mkdir(args.data_dir, exist_ok=True, parents=True)

            with open(str(Path(args.data_dir) / cached_features_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)


class PointerNetworkCodeNetTextDataset(Dataset):
    def __init__(self, tokenizer, args, mode, logger):
        self.args = args
        self.tokenizer = tokenizer
        cached_features_file = Path(args.data_dir) / f"feats_ptr_{mode}.pkl"
        filename = Path(args.data_dir) / f"{mode}-dataset.jsonl"

        if Path(cached_features_file).is_file():
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            self.examples = []
            total_num, error_num = 0, 0
            logger.info(f"Load and create features from dataset file at {filename}")
            num_lines = sum(1 for _ in open(str(filename), 'r'))
            with open(str(filename), "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines):
                    json_line = json.loads(line)
                    if len(json_line['code_tokens']) != 0: 
                        for criterion, criterion_slices in json_line["slices"].items():
                            for occurrence, slice in enumerate(criterion_slices):
                                if len(slice) != 1:
                                    total_num += 1
                                    code_tokens = json_line["code_tokens"]
                                    code_tokens = tokenizer.tokenize(" ".join(code_tokens))
                                    slice_tokens = [f"<{slice_line}>" for slice_line in slice]
                                    pointer_labels = []
                                    for tok in slice_tokens:
                                        for tok_idx, code_tok in enumerate(code_tokens):
                                            if tok == code_tok:
                                                pointer_labels.append(tok_idx)
                                                break
                                    assert len(pointer_labels) == len(slice_tokens)

                                    js = {
                                        "id": f"{json_line['id']}_{criterion}_{occurrence + 1}",
                                        "code": json_line['code'],
                                        "code_tokens": code_tokens,
                                        "pointer_labels": pointer_labels,
                                        "slice_tokens": slice_tokens,
                                        "criterion": int(criterion),
                                        "occurrence": occurrence + 1,
                                    }
                                    try:
                                        features = convert_examples_to_pointer_features(js, tokenizer)
                                        self.examples.append(features)
                                    except Exception as e:
                                        print(e)
                                        error_num += 1

            logger.warning(f"Num examples = {len(self.examples)}")
            logger.warning(f"Error num = {error_num}")
            logger.warning(f"Saving features into cached file {cached_features_file}")

            if not Path(args.data_dir).exists():
                Path.mkdir(args.data_dir, exist_ok=True, parents=True)

            with open(str(Path(args.data_dir) / cached_features_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]
        max_source_size, max_target_size = self.args.max_source_size, self.args.max_target_size

        # Encoder-Decoder for Trace Generation
        occurrence_tokens = self.tokenizer.tokenize(str(js.occurrence))
        criterion_token = self.tokenizer.tokenize(f"<{js.criterion}>")

        source_tokens = js.code_tokens[: max_source_size - 9 - len(occurrence_tokens)]
        source_tokens = ["<s>", "<encoder-decoder>", "</s>"] + source_tokens + ["</s>"] + \
                        criterion_token + ["</s>"] + occurrence_tokens + ["<mask0>", "</s>"]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_padding_length = max_source_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id for _ in range(source_padding_length)]

        target_tokens = js.pointer_labels[: max_target_size - 2]
        target_tokens = [tok + 3 for tok in target_tokens]
        target_ids = self.tokenizer.convert_tokens_to_ids(["<mask0>"]) + \
                     target_tokens + \
                     self.tokenizer.convert_tokens_to_ids(["</s>"])
        target_padding_length = max_target_size - len(target_ids)
        target_ids += [max_source_size for _ in range(target_padding_length)]

        gold_tokens = js.slice_tokens[: max_target_size - 2]
        gold_tokens = ["<mask0>"] + gold_tokens + ["</s>"]
        gold_ids = self.tokenizer.convert_tokens_to_ids(gold_tokens)
        gold_padding_length = max_target_size - len(gold_ids)
        gold_ids += [self.tokenizer.pad_token_id for _ in range(gold_padding_length)]

        return (
               torch.tensor(source_ids),
               torch.tensor(target_ids),
               torch.tensor(gold_ids),
               )


class QETextDataset(CodeNetTextDataset):
    def __init__(self, tokenizer, args, mode, logger, node_type, partial=False):
        self.args = args
        self.tokenizer = tokenizer
        if not partial:
            cached_features_file = Path(args.data_dir) / f"feats_{mode}.pkl"
            cached_qe_file = Path(args.data_dir) / f"feats_qe_{node_type}_{mode}.pkl"
        else:
            cached_features_file = Path(args.data_dir) / f"feats_{mode}_partial.pkl"
            cached_qe_file = Path(args.data_dir) / f"feats_qe_{node_type}_{mode}_partial.pkl"

        if Path(cached_qe_file).is_file():
            logger.warning(f"Loading features from cached file for qualitative evaluation {cached_qe_file}")
            with open(cached_qe_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
        else:
            logger.warning(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, 'rb') as handle1:
                input_examples = pickle.load(handle1)

            self.examples = []
            for example in input_examples:
                if not partial:
                    node_ranges = code_to_ranges(tokenizer, [node_type], code_tokens=example.code_tokens)[node_type]
                else:
                    node_ranges = code_to_ranges(tokenizer, [node_type], code=example.code)[node_type]

                for nrange in node_ranges:
                    if nrange[0] -1 < example.criterion < nrange[1] - 1:
                        self.examples.append(
                            QEInputFeatures(id=example.id,
                                            code=example.code,
                                            code_tokens=example.code_tokens,
                                            slice_tokens=example.slice_tokens,
                                            criterion=example.criterion,
                                            occurrence=example.occurrence,
                                            range=nrange,
                        ))
                        break

            logger.warning(f"Num examples for {node_type} = {len(self.examples)}")
            logger.warning(f"Saving features into cached file {cached_qe_file}")
            with open(str(Path(args.data_dir) / cached_qe_file), 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)
