'''
train, val, test split여기서
미리 edge_index 등 그래프 구성도 여기서
jarvis figshare data에서 graph-text pair를 여기서 만들어서 json으로 저장
'''
from jarvis.db.figshare import data as jdata
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
# from mendeleev import element

import pandas as pd
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from atoms2graph import AtomsToGraphs
import json
from pandarallel import pandarallel # import pandarallel
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize(progress_bar=True) # initialize pandarallel

# dataset_processed = dataset.map(preprocess_item, batched=False)

def count_elements(lst):
    '''
    Count elements in a list    
    '''
    element_count = {}
    for elem in lst:
        if elem in element_count:
            element_count[elem] += 1
        else:
            element_count[elem] = 1
    return element_count


def print_element_counts(element_counts):
    '''
    Return a string of elem counts
    '''
    text = 'The unit cell consists of '
    for elem, count in element_counts.items():
        text += f'{count} {element(elem).name} atoms, '
    text = text[:-2] + '.'
    return text


class JarvisToJson:
    def __init__(self,
                 max_neigh: int = 12,
                 radius: int = 8,
                 r_energy: bool = False,
                 r_forces: bool = False,
                 r_distances: bool = False,
                 r_edges: bool = True,
                 r_fixed: bool = True,
                 r_pbc: bool = False):
        self.a2g = AtomsToGraphs(
            max_neigh=max_neigh,
            radius=radius,
            r_energy=r_energy,
            r_forces=r_forces,
            r_distances=r_distances,
            r_edges=r_edges,
            r_fixed=r_fixed,
            r_pbc=r_pbc
        )
    
    def convert(self, db_name):
        self.data = pd.DataFrame(jdata(db_name))
        self.data['temp'] = self.data['atoms'].parallel_apply(self.generate_graph_and_text)
        temp = self.data['temp'].to_list()
        with open(f'{db_name}_data.json', 'w') as f:
            json.dump(temp, f)
        return temp

    def generate_graph_and_text(self, atoms):
        '''
        Generate graph and text for a single atomic structure
            atoms: dict
            items: dict
        '''
        jatoms = Atoms.from_dict(atoms)
        # Generate graph
        dglgraph = Graph.atom_dgl_multigraph(jatoms, compute_line_graph=False)
        item = {}
        edges = dglgraph.edges()
        item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
        item['num_nodes'] = dglgraph.num_nodes()
        item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
        item['edge_attr'] = dglgraph.edata['r'].tolist()
        
        # Generate text
        formula = jatoms.composition.reduced_formula
        crystalsystem = Spacegroup3D(jatoms).crystal_system
        # count_elems_text = print_element_counts(count_elements(jatoms.elements))

        item['y'] = f'A POSCAR of the {crystalsystem} {formula}. '
        return item
    

def main():
    dbs = ['oqmd_3d', 'aflow2', 'cod']
    converter = JarvisToJson()
    total = []
    for db in dbs:
        total += converter.convert(db)
    with open('all_data.json', 'w') as f:
        json.dump(total, f)


if __name__ == '__main__':
    main()