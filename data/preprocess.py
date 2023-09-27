'''
train, val, test split여기서
미리 edge_index 등 그래프 구성도 여기서
jarvis figshare data에서 graph-text pair를 여기서 만들어서 json으로 저장
'''
from jarvis.db.figshare import data as jdata
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
import pandas as pd
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from atoms2graph import AtomsToGraphs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# dataset_processed = dataset.map(preprocess_item, batched=False)


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
        self.data['temp'] = self.data['atoms'].apply(self.generate_graph_and_text)
        temp = self.data['temp'].to_list()
        with open(f'{db_name}.json', 'w') as f:
            json.dump(temp, f)

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
        crystalsystem = SpacegroupAnalyzer(jatoms.pymatgen_converter()).get_crystal_system()
        
        item['y'] = f'A POSCAR of {crystalsystem} {formula}'
        return item
    

def main():
    dbs = ['mp_3d_2020', 'oqmd_3d', 'aflow2', 'cod']
    converter = JarvisToJson()
    for db in dbs:
        converter.convert(db)


if __name__ == '__main__':
    main()