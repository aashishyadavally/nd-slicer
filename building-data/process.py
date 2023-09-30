import json
import jsonlines
import random
import statistics
from collections import Counter
from operator import index, itemgetter
from pathlib import Path

from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb

from transformers import RobertaTokenizer

from tree_sitter import Language, Parser

from tqdm import tqdm

from dfg_utils import (remove_comments_and_docstrings, extract_dataflow,
                       DFG_python, index_to_code_token)


PATH_TO_CODENETMUT = "codenetmut_test.json"
PATH_TO_DATASET = "../data"
TOTAL = 19541


def build_dataset(build_using, lang):
    ast_parser = get_ast_parser(lang)

    items = []
    with open(PATH_TO_CODENETMUT, 'r') as f:
        for json_line in tqdm(f):
            try:
                item_dict = json.loads(json_line)
                code = remove_comments_and_docstrings(item_dict['code'], lang)

                if build_using == 'data-flow':
                    dfg_parser = DFG_python
                    tokens, dfg, index_to_code = extract_dataflow(code, ast_parser, dfg_parser)
                    dfg = reformat_dfg(dfg)
                    variable_flow = extract_variable_flow_from_dfg(dfg, index_to_code)
                elif build_using == 'program-graph':
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

                items.append({
                    'id': item_dict['id'],
                    'code': item_dict['code'],
                    'code_tokens': item_dict['code_tokens'],
                    'trace': trace,
                    'slices': slices,
                })
            except: pass
    return items


def build_trace_with_occurrences(trace):
    trace_with_occurrences = []
    for item in trace:
        occurrence = 1
        while [item['line'], occurrence] in trace_with_occurrences:
            occurrence += 1
        trace_with_occurrences.append([item['line'], occurrence])
    return trace_with_occurrences


def extract_all_loops(parser, code):
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    loops = tree_to_loop(root_node)
    code_lines = code.split('\n')
    loops_with_vars = []
    for loop in loops:
        iterator_vars = [index_to_code_token(iterator_var, code_lines) for iterator_var in loop[-1]]
        loops_with_vars.append(loop[:-1] + (iterator_vars,))
    return loops_with_vars


def extract_all_variables(parser, code):
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    identifiers_idx = tree_to_variable_idx(root_node)
    code_lines = code.split('\n')
    identifiers = [(index_to_code_token(x, code_lines),) + x for x in identifiers_idx]
    return identifiers


def extract_dynamic_slices(criterion, trace, variable_flow):
    '''For a given slicing criterion line number, extract dynamic slices
    for all occurrences of that statement.

    Arguments:
        criterion (int):
            Criterion line number
        trace (list of dicts):
            Each item in the trace contains the line number and the program state.
        variable_flow (list of dicts):
            Each item in the variable flow contains the line number, and tuples of
            the variables and their corresponding reaching statements.

    Returns:
        slices (list):

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


def extract_variable_flow_from_dfg(dfg, index_to_code):
    variable_flow = {}
    for var, edges in dfg.items():
        for edge in edges:
            cur_statement = index_to_code[edge['tokenId']][0][0]
            from_statements = [(var, index_to_code[x][0][0]) for x in edge['from']]
            if cur_statement in variable_flow:
                temp = variable_flow[cur_statement]
                variable_flow[cur_statement] = temp + from_statements
            else:
                variable_flow[cur_statement] = from_statements
    
    for cur_statement, from_statements in variable_flow.items():
        variable_flow[cur_statement] = sorted(list(from_statements), key=itemgetter(1))

    return variable_flow


def extract_variable_flow_from_pdg(code):
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


def get_ast_parser(lang):
    if not Path(f'build/ts-languages.so').exists():
        Language.build_library(
            # Store the library in the `build` directory
            'build/ts-languages.so',
            # Include one or more languages
            [f'build/tree-sitter-{lang}'],
        )

    parser = Parser()
    language = Language('build/ts-languages.so', lang)
    parser.set_language(language)
    return parser


def reformat_dfg(dfg):
    new_dfg = {}
    for item in dfg:
        if 'comesFrom' not in item:
            continue
        from_edges = [edge for edge in item[-1] if edge != item[1]]
        if from_edges == []:
            continue
        if item[0] not in new_dfg:
            new_dfg[item[0]] = [{'tokenId': item[1], 'from': from_edges}]
        else:
            new_dfg[item[0]] += [{'tokenId': item[1], 'from': from_edges}]
    return new_dfg


def tree_to_loop(root_node):
    if root_node.type in ['for_statement', 'while_statement']:
        iterator_var = []
        for child in root_node.children:
            if len(child.children) == 0 and child.type == 'identifier':
                iterator_var += [(child.start_point, child.end_point)]
        return [(root_node.start_point, root_node.end_point, iterator_var)]
    else:
        loop_ranges = []
        for child in root_node.children:
            loop_ranges += tree_to_loop(child)
        return loop_ranges


def tree_to_variable_idx(root_node):
    if len(root_node.children) == 0 and root_node.type == 'identifier':
        return [(root_node.start_point, root_node.end_point)]
    else:
        identifiers_idx = []
        for child in root_node.children:
            identifiers_idx += tree_to_variable_idx(child)
        return identifiers_idx


if __name__ == '__main__':
    Path(PATH_TO_DATASET).mkdir(exist_ok=True)
    path_to_datafile = Path(PATH_TO_DATASET) / 'full-dataset.jsonl'

    try:
        with jsonlines.open(str(path_to_datafile), 'r') as f:
            items = [_dict for _dict in f]

        with jsonlines.open(str(Path(PATH_TO_DATASET) / 'train-dataset.jsonl'), 'r') as f:
            train_items = [_dict for _dict in f]

        with jsonlines.open(str(Path(PATH_TO_DATASET) / 'val-dataset.jsonl'), 'r') as f:
            val_items = [_dict for _dict in f]

        with jsonlines.open(str(Path(PATH_TO_DATASET) / 'test-dataset.jsonl'), 'r') as f:
            test_items = [_dict for _dict in f]

    except FileNotFoundError:
        items = build_dataset(build_using='program-graph', lang='python')
        with open(str(path_to_datafile), 'w') as f:
            f.write('\n'.join(map(json.dumps, items)))

        random.seed(42)
        random.shuffle(items)
        num_train, num_eval = int(0.8 * len(items)), int(0.1 * len(items))
        train_items, val_items, test_items = (items[:num_train],
                                              items[num_train: num_train+num_eval],
                                              items[num_train+num_eval:])
        with open(str(Path(PATH_TO_DATASET) / 'train-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, train_items)))

        with open(str(Path(PATH_TO_DATASET) / 'val-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, val_items)))

        with open(str(Path(PATH_TO_DATASET) / 'test-dataset.jsonl'), 'w') as f:
            f.write('\n'.join(map(json.dumps, test_items)))

    print()
    print('*** Dataset statistics ***')
    print(f'  Total number of code examples: {TOTAL}')
    print(f'  Number of code examples after skipping: {len(items)}')
