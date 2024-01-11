# import numpy as np
import requests
import pandas as pd
import numpy as np
# from jarvis.core.atoms import Atoms
# from jarvis.core.graphs import Graph
from joblib import Parallel, delayed
from tqdm import tqdm


def pubchem_request(chemical_name, key='name', task=None, max_records=1, structure_type=None):
    '''
    If task is None, return just result of compound name 
    
    input: chemical_name, key, task, max_records, structure_type
        chemical_name: str
        key: str
        task: str
        max_records: int
        structure_type: str
    output: list of dict
    '''
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/'
    if task == 'fastsimilarity_2d':
        url += task + '/'
    url += f'{key}/{chemical_name}/property/CanonicalSMILES,InChI,IUPACName/JSON?MaxRecords={max_records}'
    if structure_type == '3d':
        url += '&record_type=3d'
    response = requests.get(url)
    return response.json()['PropertyTable']['Properties']


def pubchem_request_synonym(chemical_name):
    response = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}synonyms/JSON')
    return response.json()['InformationList']['Information'][0]['Synonym']


def search_mat_in_paper_return_paragraphs(paper_df, mat, paper_max=10):
    '''
    one to many function. one material to many papers
    input: paper_df, mat
        paper_df: dataframe of papers
        mat: material name
        paper_max: max number of papers to search
    output: list of paragraphs, list of dois
        paragraphs: list of paragraphs containing the material name
        dois: list of dois of the papers containing the material name
    '''
    # First check if the formula is in the text
    paper_df['query'] = paper_df['text'].apply(lambda x: mat in x)

    # select papers max number of papers to search
    true_rows = paper_df[paper_df['query']]
    selected_rows = true_rows if len(true_rows) <= paper_max else true_rows.sample(n=paper_max) 

    # return paragraphs and dois
    text_series = selected_rows['text'].apply(lambda x: [p for p in x.split('\n') if mat in p]).to_list()  # list of paragraph including keyword
    doi_series = selected_rows['doi'].to_list()
    return text_series, doi_series


def jarvis_atoms2graph(input_item):
    '''
    input: pandas dataframe rows
    output: graph in dict format

    Convert atoms column in parquet file to graphs
    Graphs contain
        edge_index: list of two lists of int
        edge_attr: list of float
        num_nodes: int
        node_feat: list of list of float
        y: list of int
    '''
    if 'atoms' in input_item.keys():
        jatoms = Atoms.from_dict(input_item['atoms'])
    elif 'cif' in input_item.keys():
        with open('temp.cif', 'w') as f:
            f.write(input_item['cif'])
        jatoms = Atoms.from_cif('temp.cif')
    dglgraph = Graph.atom_dgl_multigraph(jatoms, compute_line_graph=False, atom_features='cgcnn')
    item = {}
    edges = dglgraph.edges()
    item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
    item['edge_attr'] = dglgraph.edata['r'].tolist()
    item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
    item['y'] = [0]
    return item


def json_atoms_to_graphs(json_file):
    '''
    Convert atoms column in json file to graphs
    Graphs contain
        edge_index: list of two lists of int
        edge_attr: list of float
        num_nodes: int
        node_feat: list of list of float
        y: list of int
    '''
    # Load parquet file
    df = pd.read_json(json_file)
    # Divide parquet file into chunks
    chunks = np.array_split(df, 12)
    def json_atoms_to_graphs_chunk(chunk):
        '''
        input: chunk of parquet file dataframe
        output: list of graphs
        '''
        result = []
        for atom in tqdm(chunk.iloc):
            item = jarvis_atoms2graph(atom)
            result.append(item)
        return result
    # Parallelize
    results = Parallel(n_jobs=12)(delayed(json_atoms_to_graphs_chunk)(chunk) for chunk in chunks)
    results = sum(results, [])
    return results    