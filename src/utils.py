

from tree_sitter import Language, Parser, Query
import logging
import os
import hashlib
from magika import Magika
from pathlib import Path
from scipy.stats import entropy
from collections import Counter
import networkx as nx
from tree_sitter import Node
import pyfiglet
from tree_sitter import Language, Parser, Query
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_powershell as tspowershell
import tree_sitter_html as tshtml
import tree_sitter_php as tsphp



logger = logging.getLogger(__name__) 


LANGUAGE_MAP = {
    "python": tspython.language(),
    "javascript": tsjavascript.language(),
    "powershell": tspowershell.language(),
    "php": tsphp.language_php(),
    "html": tshtml.language(),
}


def generate_rn_script_ascii_art(font="standard", width=80):
    try:
        result = pyfiglet.figlet_format("r.n.Scripts", font=font, width=width)
        return result
    except Exception as e:
        print(f"Error generating ASCII art: {e}")
        return None


def shannon_entropy_scipy(data: bytes) -> float:
    freq = Counter(data)
    probabilities = [count / len(data) for count in freq.values()]
    return entropy(probabilities, base=2)


def sha256_hash(data: bytes) -> str:
    """Generate the SHA-256 hash of the input buffer."""
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()


def write_to_file(filename, content):
    """
    Writes the given content to a file in the current working directory.

    Args:
        filename (str): The name of the file to create/overwrite.
        content (str): The content to write to the file.
    """
    try:
        with open(filename, "w") as f:  
            f.write(content)
        print(f"\n[+] - Successfully wrote to file: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def read_file(filepath: str) -> str | None:
    """
    Reads the content of a file, trying various encodings.
    """
    encodings_to_try = [
        "utf-8",
        "utf-16",       
        "utf-16-le",    
        "utf-16-be",
        "cp1252",       
        "latin-1"       
    ]

    for encoding in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
                logger.info(f"Successfully read file '{filepath}' with encoding: {encoding}")
                return content
        except UnicodeDecodeError:
            logger.debug(f"Failed to read file '{filepath}' with encoding: {encoding} (UnicodeDecodeError)")
            continue 
        except Exception as e:
            
            logger.error(f"Error reading file '{filepath}' with encoding '{encoding}': {e}")
            return None 

    logger.error(f"Failed to read file '{filepath}' with any of the tried encodings.")
    return None


def magika_type(file_input):
    magika_instance = Magika()

    file_path = Path(file_input)

    try:
        result = magika_instance.identify_path(file_path)
        if result:
            return result.output
        else:
            print(f"Could not identify the file type for '{file_input}'")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def node_to_dict(node, code):
    """Recursively convert a TreeSitter node to a dictionary."""
    return {
        "type": node.type,
        "text": code[node.start_byte:node.end_byte].decode(),  
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": (node.start_point.row, node.start_point.column),
        "end_point": (node.end_point.row, node.end_point.column),
        "children": [node_to_dict(child, code) for child in node.children]
    }

def generate_ast(ct_type, code):
    try:
        language = LANGUAGE_MAP.get(ct_type)
        if language is None:
            
            logger.warning(f"Unsupported language type for analysis: {ct_type}")
            return None, None

        LANGUAGE = Language(language)
        parser = Parser(LANGUAGE)
        tree = parser.parse(code)
        root_node = tree.root_node
        node_dict = node_to_dict(root_node, code)
        return tree, node_dict
    except Exception as e:
        
        logger.error(f"An error occurred during code analysis for type '{ct_type}': {e}")
        return None, None

def convert_ast_to_graph(ast_root):
    graph = nx.DiGraph()

    def traverse(node, parent_id=None):
        node_id = id(node)
        graph.add_node(node_id, label=node.type, start_byte=node.start_byte, end_byte=node.end_byte)  

        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for child in node.children:
            traverse(child, node_id)

    traverse(ast_root)
    return graph


def get_all_shortest_path_edges_subgraph(G: nx.Graph, weight_attribute: str = None) -> nx.Graph:
    """
    Creates a new graph containing only edges that lie on at least one shortest path
    between any two nodes in the original graph. This helps in 'noise reduction'
    by keeping only the most direct or 'efficient' connections.

    Args:
        G: The input NetworkX graph (nx.Graph or nx.DiGraph).
        weight_attribute: The name of the edge attribute to use as weight for
                          shortest path calculations. If None, unweighted (hop count)
                          shortest paths are considered. Weights should represent
                          'cost' or 'distance' where lower is better.

    Returns:
        A new NetworkX graph of the same type as G, containing only the nodes
        and edges that are part of at least one shortest path.
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        logger.error("Input graph G must be a NetworkX Graph or DiGraph.")
        return G.__class__()  

    trimmed_G = G.__class__()
    trimmed_G.add_nodes_from(G.nodes(data=True))
    shortest_path_edges_to_add = set()
    node_list = list(G.nodes())
    for i, source in enumerate(node_list):
        
        
        targets_to_consider = node_list
        if not G.is_directed():
            targets_to_consider = node_list[i:]  

        for target in targets_to_consider:
            if source == target:
                continue  

            try:
                
                
                paths_iterator = nx.all_shortest_paths(G, source, target, weight=weight_attribute)

                for path_nodes in paths_iterator:
                    
                    for k in range(len(path_nodes) - 1):
                        u, v = path_nodes[k], path_nodes[k + 1]

                        
                        if G.is_directed():  
                            shortest_path_edges_to_add.add((u, v))
                        else:  
                            shortest_path_edges_to_add.add(tuple(sorted((u, v))))
            except nx.NetworkXNoPath:
                
                continue
            except Exception as e:
                
                logger.error(f"Error finding shortest path between {source} and {target}: {e}")
                continue

    
    for u, v in shortest_path_edges_to_add:
        
        if G.has_edge(u, v):
            trimmed_G.add_edge(u, v, **G.get_edge_data(u, v))
        
        elif not G.is_directed() and G.has_edge(v, u):
            trimmed_G.add_edge(v, u, **G.get_edge_data(v, u))

    logger.info(
        f"Graph trimming applied. Original edges: {G.number_of_edges()}, Trimmed edges: {trimmed_G.number_of_edges()}")
    return trimmed_G


def trim_ast_json_dict(node_dict: dict) -> dict | list | None:
    """
    Recursively processes an AST node (dictionary representation) to flatten
    out redundant intermediate nodes that primarily serve as syntactic wrappers.
    This reduces the depth and verbosity of the AST, making it more concise.

    Nodes with types in `FILTER_OUT_TYPES` are removed, and their children are
    effectively promoted to their position in the hierarchy.

    Args:
        node_dict: The dictionary representation of an AST node. Must have 'type' and 'children' keys.

    Returns:
        A new, simplified dictionary representation of the AST node.
        If a node is flattened and has multiple children, a list of its children
        (after their own recursive filtering) is returned.
        If a node is a leaf (no children) and its type is a purely decorative/punctuation type,
        then `None` is returned, effectively removing it.
    """

    FILTER_OUT_TYPES = {
        "logical_expression",
        "bitwise_expression",
        "comparison_expression",
        "additive_expression",
        "multiplicative_expression",
        "format_expression",
        "range_expression",
        "unary_expression",
        "expression_with_unary_operator",
        "cast_expression",
        "parenthesized_expression",
        "left_assignment_expression",
        "statement_list",  
        "script_block_body"  
    }

    LEAF_DECORATIVE_TYPES = {
        "(", ")", "[", "]", "{", "}", ",", ";", "=", ":",
        " ",  
        "++", "--",  
        "&", "|", "^", "~", "<<", ">>",  
        "\\r", "\\n"  
    }

    if not node_dict.get('children'):
        if node_dict['type'] in LEAF_DECORATIVE_TYPES:
            return None  
        return node_dict  

    current_children_after_filtering = []
    for child in node_dict['children']:
        processed_child = trim_ast_json_dict(child)  
        if processed_child is None:
            
            continue
        elif isinstance(processed_child, list):
            
            current_children_after_filtering.extend(processed_child)
        else:
            
            current_children_after_filtering.append(processed_child)
    node_type = node_dict['type']
    if node_type in FILTER_OUT_TYPES:
        if current_children_after_filtering:
            return current_children_after_filtering  
        else:
            return None

    new_node_dict = node_dict.copy()  
    new_node_dict['children'] = current_children_after_filtering

    return new_node_dict