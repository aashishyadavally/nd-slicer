'''Contains utility functions for module.
'''
import re
from operator import itemgetter
from pathlib import Path

from tree_sitter import Language, Parser

from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb


PATH_TO_BUILD = '../data/build'
REGEX = '^[0-9]+$'


def get_ast_parser(path_to_build):
    '''Builds and returns tree-sitter's AST parser.

    Arguments:
        path_to_build (str): Path to directory where the tree-sitter languages need to be built.

    Returns:
        parser (tree_sitter.Parser): AST parser.
    '''
    if not Path(f'{path_to_build}/ts-languages.so').exists():
        Language.build_library(
            # Store the library in the `build` directory
            f'{path_to_build}/ts-languages.so',
            # Include one or more languages
            [f'{path_to_build}/tree-sitter-python'],
        )
    parser = Parser()
    language = Language(f'{path_to_build}/ts-languages.so', 'python')
    parser.set_language(language)
    return parser


def code_to_method_calls(root_node):
    '''Recursively extracts all method calls from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.

    Returns:
        method_calls (list): List of all method calls in the code.
    '''
    if root_node.type == 'identifier' and root_node.parent.type == 'call':
        if root_node.start_point[0] == root_node.end_point[0]:
            return [{
                'line': root_node.start_point[0],
                'start_idx': root_node.start_point[1],
                'end_idx': root_node.end_point[1]
            }]
    else:
        method_calls = []
        for child in root_node.children:
            method_calls += code_to_method_calls(child)
        return method_calls


def code_to_method_definitions(root_node):
    '''Recursively extracts all method definitions from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.

    Returns:
        definitions (list): List of all method definitions in the code.
    '''
    if root_node.type == 'identifier' and root_node.parent.type == 'function_definition':
        if root_node.start_point[0] == root_node.end_point[0]:
            return [{
                'start_line': root_node.start_point[0],
                'end_line': root_node.parent.end_point[0],
                'start_idx': root_node.start_point[1],
                'end_idx': root_node.end_point[1]
            }]
    else:
        definitions = []
        for child in root_node.children:
            definitions += code_to_method_definitions(child)
        return definitions


def code_to_ranges(tokenizer, node_types, code=None, code_tokens=None):
    '''Recursively extracts ranges of all nodes belonging to ``node_types`` in the AST.

    Arguments:
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.
        node_types (list): List of node types to consider during AST traversal.
        code (str or None): Source code.
        code_tokens (list or None): Tokenized code tokens.

    Returns:
        node_ranges (list): List of ranges of select nodes in AST.
    '''
    if not code:
        code = tokens_to_string(code_tokens, tokenizer)
    ast_parser = get_ast_parser(PATH_TO_BUILD)
    tree = ast_parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    node_ranges = {}
    for node_type in node_types:
        node_ranges[node_type] = tree_to_node_range(root_node, node_type)
    return node_ranges


def code_to_returns(root_node):
    '''Recursively extracts all return statements from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.

    Returns:
        method_calls (list): List of all return statements in the code.
    '''
    if root_node.type == 'return_statement':
        return [{'line': root_node.end_point[0]}]
    else:
        method_calls = []
        for child in root_node.children:
            method_calls += code_to_method_calls(child)
        return method_calls


def extract_all_variables(parser, code):
    '''Extract all variables in input ``code``.

    Arguments:
        parser (tree_sitter.Parser): AST parser.
        code (str): Source code.
    
    Returns:
        identifiers (list): List of all variables in ``code``.
    '''
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    identifiers_idx = tree_to_variable_idx(root_node)
    code_lines = code.split('\n')
    identifiers = [(index_to_code_token(x, code_lines),) + x for x in identifiers_idx]
    return identifiers


def extract_dynamic_slice(criterion, trace, occurrence, variable_flow):
    '''Extract dynamic slice from full execution trace and PDG using Agrawal's algorithm.

    Arguments:
        criterion (int): Slicing criterion.
        trace (list): Execution trace.
        occurrence (int): Executing iteration of slicing criterion.
        variable_flow (list): Flow of variables in the PDG.

    Returns:
        (list): Dynamic slice corresponding to ``criterion``.
    '''
    if criterion not in trace:
        return []

    criterion_ctr = 0
    for i, item in enumerate(trace):
        if item == criterion:
            criterion_ctr += 1
        if criterion_ctr == occurrence:
            end_idx = i
            break
    reduced_trace = trace[: end_idx]
    if criterion in variable_flow:
        occurrence_slice = [criterion]
        reaching_statements = [x[1] for x in variable_flow[criterion]]
        # Traverse reduced trace in reverse order.
        for item in reversed(reduced_trace):
            if item in reaching_statements:
                occurrence_slice.append(item)
                if item in variable_flow:
                    reaching_statements = [x[1] for x in variable_flow[item]]
                else:
                    continue
            else:
                continue
        return occurrence_slice
    else:
        return [criterion]


def extract_dynamic_slice_with_reduced_trace(criterion, reduced_trace, variable_flow):
    '''Extract dynamic slice from reduced execution trace (i.e., when directly given a subset of
    full execution trace, until the corresponding occurrence of the slicing criterion) and PDG
    using Agrawal's algorithm.

    Arguments:
        criterion (int): Slicing criterion.
        reduced_trace (list): Reduced execution trace.
        variable_flow (list): Flow of variables in the PDG.

    Returns:
        (list): Dynamic slice corresponding to ``criterion``.
    '''
    if criterion in variable_flow:
        occurrence_slice = [criterion]
        reaching_statements = [x[1] for x in variable_flow[criterion]]
        # Traverse reduced trace in reverse order.
        for item in reversed(reduced_trace):
            if item in reaching_statements:
                occurrence_slice.append(item)
                if item in variable_flow:
                    reaching_statements = [x[1] for x in variable_flow[item]]
                else:
                    continue
            else:
                continue
        return occurrence_slice
    else:
        return [criterion]


def extract_import_lines(parser, code):
    '''Extract line numbers of all import statements in input ``code``.

    Arguments:
        parser (tree_sitter.Parser): AST parser.
        code (str): Source code.
    
    Returns:
        import_lines (list): List of line numbers of all import statements in ``code``.
    '''
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    import_lines = tree_to_import_lines(root_node)
    return import_lines


def extract_reaching_statements(variable, criterion, code):
    '''Extract all statements that can be reached by a variable along the PDG.

    Arguments:
        variable (str): Variable in ``code``.
        criterion (int): Slicing criterion.
        code (str): Source code.

    Returns:
        relevant_statements (list): List of reaching statements.        
    '''
    variable_flow = extract_variable_flow_from_pdg(code)
    relevant_statements = []

    for to_edges in variable_flow.values():
        for var, to_line in to_edges:
            if var == variable and to_line not in relevant_statements and to_line != criterion:
                relevant_statements.append(to_line)
    return sorted(relevant_statements)


def extract_trace(complete_trace, extract_states=False):
    '''Extract execution trace from CodeExecutor dataset.

    Arguments:
        complete_trace (list): Original trace as in CodeExecutor dataset, which contains
            executed line number, and corresponding program states.
        extract_states (bool): If True, extract program states as well.

    Returns:
        trace (list): Extracted execution trace.
    '''
    trace = []
    for trace_line in complete_trace:
        line_output = trace_line.split('<state>')[0]
        if line_output.startswith('<output>'):
            continue

        line_number = int(line_output.split('<line>')[1].strip()
                                     .split('<')[1].split('>')[0])

        state_output = trace_line.split('<state>')[1].split('</state>')[0]
        if state_output.strip() == '':
            state = []
        else:
            state = [x.strip() for x in state_output.split('<dictsep>')]
            state = [(x.split(':')[0].strip(), x.split(':')[1].strip())
                     for x in state]

        trace_item = {'line': line_number}
        if extract_states:
            trace_item['state'] = state
        trace.append(trace_item)
    return trace


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
        variable_flow[cur_statement] = sorted(
            list(from_statements), key=itemgetter(1))

    return variable_flow


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]: end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += " " + code[i]
        s += " " + code[end_point[0]][: end_point[1]]
    return s


def string_to_tokens(code, tokenizer):
    '''Convert code string to code tokens, introducing '<indent>' and '<dedent>' tokens
    to indicate entering and exiting scope of code blocks.

    Arguments:
        code (str): Source code.
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.

    Returns:
        code_tokens (list): Tokenized code tokens.
    '''
    code_tokens = []
    previous_line_spaces = 0
    for line_number, line in enumerate(code.split('\n')):
        line_tokens = [f'<{line_number}>']
        num_spaces = len(line) - len(line.lstrip())
        if num_spaces > previous_line_spaces:
            line_tokens += ['<indent>' for _ in range(
                (num_spaces - previous_line_spaces) // 4)]
        elif num_spaces < previous_line_spaces:
            line_tokens += ['<dedent>' for _ in range(
                (previous_line_spaces - num_spaces) // 4)]
        previous_line_spaces = num_spaces
        line_tokens += tokenizer.tokenize(line.lstrip())
        code_tokens += line_tokens
    return code_tokens


def tokens_to_string(original_tokens, tokenizer):
    '''Combines code tokens into source code, replacing '<indent>' and '<dedent>' tokens with
    corresponding scoping as required for the code blocks.

    Arguments:
        original_tokens (list): Code tokens.
        tokenizer (transformers.RobertaTokenizer): Input tokenizer.

    Returns:
        (str): Constructed source code.
    '''
    new_tokens, previous_tok_idx = [], 0
    for tok_idx, tok in enumerate(original_tokens):
        if tok.startswith('<') and tok.endswith('>') and re.search(REGEX, tok[1:-1]):
            new_tokens.append(original_tokens[previous_tok_idx + 1: tok_idx])
            previous_tok_idx = tok_idx
    new_tokens.append(original_tokens[previous_tok_idx + 1:])

    code_lines, previous_indent = [], ''
    for line_code_tokens in new_tokens:
        num_indent, num_dedent = 0, 0
        for tok in line_code_tokens:
            if tok == '<indent>':
                num_indent += 1
            if tok == '<dedent>':
                num_dedent += 1
        if num_indent > 0:
            previous_indent += '    ' * num_indent
        if num_dedent > 0:
            previous_indent = previous_indent[:-(4 * num_dedent)]
        line_string = tokenizer.decode(tokenizer.convert_tokens_to_ids(
            line_code_tokens[num_indent+num_dedent:]))
        code_lines.append(previous_indent + line_string)
    return '\n'.join(code_lines)


def tree_to_import_lines(root_node):
    '''Recursively extracts import statement information from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.

    Returns:
        import_lines (list): List of all import statements in the code.
    '''
    if root_node.type in [
        'import_statement', 'import_from_statement', 'future_import_statement', 'aliased_import'
    ]:
        return [root_node.start_point[0]]
    else:
        import_lines = []
        for child in root_node.children:
            import_lines += tree_to_import_lines(child)
        return import_lines


def tree_to_node_range(root_node, node_type):
    '''Recursively extracts node ranges of specific ``node_type`` from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.
        node_type (str): AST node type, as in tree_sitter's grammar.

    Returns:
        node_ranges (list): List of all node ranges of specific ``node_type`` in the code.
    '''
    if root_node.type == node_type:
        return [(root_node.start_point[0], root_node.end_point[0])]
    else:
        node_ranges = []
        for child in root_node.children:
            node_ranges += tree_to_node_range(child, node_type)
        return node_ranges


def tree_to_variable_idx(root_node):
    '''Recursively extracts all variables' information from given source code.

    Arguments:
        root_node (tree_sitter.Node): Root node of code AST.

    Returns:
        identifiers_idx (list): List of all identifiers in the code.
    '''
    if len(root_node.children) == 0 and root_node.type == 'identifier':
        return [(root_node.start_point, root_node.end_point)]
    else:
        identifiers_idx = []
        for child in root_node.children:
            identifiers_idx += tree_to_variable_idx(child)
        return identifiers_idx
