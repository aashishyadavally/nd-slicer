'''Main script for building the dataset from scratch. Skip if ``{train|val|test}-dataset.json``
files have already been downloaded from Google Drive or Zenodo.

Preqrequistes:
--------------
This script expects the presence of ``codenetmut_test.json`` file in the current
directory. If not already present, download from: https://zenodo.org/records/8062703

Usage:
------
(autoslicer) $ python process.py
'''
import json
import random
import tokenize
from collections import Counter
from io import StringIO
from operator import itemgetter
from pathlib import Path

import jsonlines

from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb

from tqdm import tqdm


PATH_TO_CODENETMUT = "codenetmut_test.json"
TOTAL = 19541


def remove_comments_and_docstrings(source):
    '''Remove comments and docstrings from a source code string.

    Arguments:
        source (str): Input code.
    
    Returns:
        (str): Processed code.
    '''
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno, last_col = -1, 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string = tok[0], tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]

        if start_line > last_lineno: last_col = 0
        if start_col > last_col: out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT: pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator.
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0: out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    temp = []
    for x in out.split('\n'):
        if x.strip() != "": temp.append(x)
    return '\n'.join(temp)


def build_dataset():
    '''Builds the dataset from scratch. Note that this can be time consuming, as it
    involves creating the PDGs for each program in the dataset using ``python_graphs``.

    Returns:
        all_items (list): List of ``dict`` items, each corresponding to a data instance.
    '''
    all_items = []
    with open(PATH_TO_CODENETMUT, 'r') as f:
        for json_line in tqdm(f):
            try:
                item_dict = json.loads(json_line)
                code = remove_comments_and_docstrings(item_dict['code'], lang)
                variable_flow = extract_variable_flow_from_pdg(code)

                trace = []
                for trace_line in item_dict['trace']:
                    line_output = trace_line.split('<state>')[0]
                    if line_output.startswith('<output>'):
                        continue

                    line_number = int(line_output.split('<line>')[1].strip()\
                                                .split('<')[1].split('>')[0])
                    state_output = trace_line.split('<state>')[1].split('</state>')[0]
                    if state_output.strip() == '':
                        state = []
                    else:
                        state = [x.strip() for x in state_output.split('<dictsep>')]
                        state = [(x.split(':')[0].strip(), x.split(':')[1].strip()) for x in state]

                    trace.append({
                        'line': line_number,
                        'state': state,
                    })

                slices = {}
                num_lines = len(item_dict['code'].split('\n'))
                for criterion in range(num_lines):
                    criterion_slices = extract_dynamic_slices(criterion, trace, variable_flow)
                    if criterion_slices:
                        slices[criterion] = criterion_slices

                all_items.append({
                    'id': item_dict['id'],
                    'code': item_dict['code'],
                    'code_tokens': item_dict['code_tokens'],
                    'trace': trace,
                    'slices': slices,
                })
            except: pass
    return all_items


def build_trace_with_occurrences(trace):
    '''Extract tuples of executing statement identifier and its corresponding
    occurrence in the execution trace.

    Arguments:
        trace (list): List of statement identifiers.
    
    Returns:
        trace_with_occurrences (list): List of tuples, each containing the statement
        identifier, and corresponding occurrence in input ``trace``.
    '''
    trace_with_occurrences = []
    for item in trace:
        occurrence = 1
        while [item['line'], occurrence] in trace_with_occurrences:
            occurrence += 1
        trace_with_occurrences.append([item['line'], occurrence])
    return trace_with_occurrences


def extract_dynamic_slices(criterion, trace, variable_flow):
    '''For a given slicing criterion line number, extract dynamic slices
    for all occurrences of that statement.

    Arguments:
        criterion (int): Criterion line number
        trace (list of dicts): Each item in the trace contains the line number
            and the program state.
        variable_flow (list of dicts): Each item in the variable flow contains
            the line number, and tuples of the variables and their corresponding
            reaching statements.

    Returns:
        slices (list): List of ground-truth dynamic slices for all occurrences of
            the slicing criterion in the trace.

    '''
    trace_with_occurrences = build_trace_with_occurrences(trace)
    num_of_occurrences = dict(Counter([item['line'] for item in trace]))

    if criterion not in num_of_occurrences:
        return []

    slices = []
    for occurrence in range(1, num_of_occurrences[criterion] + 1):
        for i, item in enumerate(trace_with_occurrences):
            if item == [criterion, occurrence]:
                end_idx = i
                break
        reduced_trace = [x[0] for _id, x in enumerate(trace_with_occurrences) if _id < end_idx]
        if criterion in variable_flow:
            occurrence_slice = [criterion]
            reaching_statements = [x[1] for x in variable_flow[criterion]]
            # Traverse reduced trace in reverse order.
            for item in reversed(reduced_trace):
                if item in reaching_statements:
                    occurrence_slice.append(item)
                    if item in variable_flow:
                        reaching_statements = [x[1] for x in variable_flow[item]]
                    else: continue
                else: continue
            slices.append(occurrence_slice)
        else:
            slices.append([criterion])
    return slices


def extract_variable_flow_from_pdg(code):
    '''Extract flows of variables along the PDG.

    Arguments:
        code (str): Input code.
    
    Returns:
        variable_flow (dict): Flows of variables along the PDG.
    '''
    graph = program_graph.get_program_graph(code)

    variable_flow = {}
    for edge in graph.edges:
        if edge.type in [pb.EdgeType.LAST_READ, pb.EdgeType.LAST_WRITE]:
            from_line = graph.get_node(edge.id1).ast_node.lineno - 1
            var_from = graph.get_node(edge.id1).ast_node.id
            to_line = graph.get_node(edge.id2).ast_node.lineno - 1
            var_to = graph.get_node(edge.id2).ast_node.id
            assert var_from == var_to

            if from_line == to_line:
                continue
            if from_line in variable_flow:
                temp = variable_flow[from_line]
                variable_flow[from_line] = temp + [(var_from, to_line)]
            else:
                variable_flow[from_line] = [(var_from, to_line)]

    for cur_statement, from_statements in variable_flow.items():
        variable_flow[cur_statement] = sorted(list(from_statements), key=itemgetter(1))

    return variable_flow


if __name__ == '__main__':
    path_to_datafile = Path.cwd() / 'full-dataset.jsonl'

    try:
        with jsonlines.open(str(path_to_datafile), 'r') as f:
            items = [_dict for _dict in f]

        with jsonlines.open(str(Path.cwd() / 'train-dataset.jsonl'), 'r') as f:
            train_items = [_dict for _dict in f]

        with jsonlines.open(str(Path.cwd() / 'val-dataset.jsonl'), 'r') as f:
            val_items = [_dict for _dict in f]

        with jsonlines.open(str(Path.cwd() / 'test-dataset.jsonl'), 'r') as f:
            test_items = [_dict for _dict in f]

    except FileNotFoundError:
        items = build_dataset()
        with open(str(path_to_datafile), 'w') as f:
            f.write('\n'.join(map(json.dumps, items)))

        random.seed(42)
        random.shuffle(items)
        num_train, num_eval = int(0.8 * len(items)), int(0.1 * len(items))
        train_items, val_items, test_items = (items[:num_train],
                                              items[num_train: num_train+num_eval],
                                              items[num_train+num_eval:])
        with open(str(Path.cwd() / 'train-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, train_items)))

        with open(str(Path.cwd() / 'val-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, val_items)))

        with open(str(Path.cwd() / 'test-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, test_items)))
