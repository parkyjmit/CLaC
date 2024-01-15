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
import numpy as np
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from atoms2graph import AtomsToGraphs
import json
from pandarallel import pandarallel 
from joblib import Parallel, delayed
from tqdm import tqdm

# tqdm.pandas()
# pandarallel.initialize(progress_bar=True) # initialize pandarallel

# dataset_processed = dataset.map(preprocess_item, batched=False)

periodic_table = {'H': 'Hydrogen',
                'He': 'Helium',
                'Li': 'Lithium',
                'Be': 'Beryllium',
                'B': 'Boron',
                'C': 'Carbon',
                'N': 'Nitrogen',
                'O': 'Oxygen',
                'F': 'Fluorine',
                'Ne': 'Neon',
                'Na': 'Sodium',
                'Mg': 'Magnesium',
                'Al': 'Aluminum',
                'Si': 'Silicon',
                'P': 'Phosphorus',
                'S': 'Sulfur',
                'Cl': 'Chlorine',
                'Ar': 'Argon',
                'K': 'Potassium',
                'Ca': 'Calcium',
                'Sc': 'Scandium',
                'Ti': 'Titanium',
                'V': 'Vanadium',
                'Cr': 'Chromium',
                'Mn': 'Manganese',
                'Fe': 'Iron',
                'Co': 'Cobalt',
                'Ni': 'Nickel',
                'Cu': 'Copper',
                'Zn': 'Zinc',
                'Ga': 'Gallium',
                'Ge': 'Germanium',
                'As': 'Arsenic',
                'Se': 'Selenium',
                'Br': 'Bromine',
                'Kr': 'Krypton',
                'Rb': 'Rubidium',
                'Sr': 'Strontium',
                'Y': 'Yttrium',
                'Zr': 'Zirconium',
                'Nb': 'Niobium',
                'Mo': 'Molybdenum',
                'Tc': 'Technetium',
                'Ru': 'Ruthenium',
                'Rh': 'Rhodium',
                'Pd': 'Palladium',
                'Ag': 'Silver',
                'Cd': 'Cadmium',
                'In': 'Indium',
                'Sn': 'Tin',
                'Sb': 'Antimony',
                'Te': 'Tellurium',
                'I': 'Iodine',
                'Xe': 'Xenon',
                'Cs': 'Cesium',
                'Ba': 'Barium',
                'La': 'Lanthanum',
                'Ce': 'Cerium',
                'Pr': 'Praseodymium',
                'Nd': 'Neodymium',
                'Pm': 'Promethium',
                'Sm': 'Samarium',
                'Eu': 'Europium',
                'Gd': 'Gadolinium',
                'Tb': 'Terbium',
                'Dy': 'Dysprosium',
                'Ho': 'Holmium',
                'Er': 'Erbium',
                'Tm': 'Thulium',
                'Yb': 'Ytterbium',
                'Lu': 'Lutetium',
                'Hf': 'Hafnium',
                'Ta': 'Tantalum',
                'W': 'Tungsten',
                'Re': 'Rhenium',
                'Os': 'Osmium',
                'Ir': 'Iridium',
                'Pt': 'Platinum',
                'Au': 'Gold',
                'Hg': 'Mercury',
                'Tl': 'Thallium',
                'Pb': 'Lead',
                'Bi': 'Bismuth',
                'Po': 'Polonium',
                'At': 'Astatine',
                'Rn': 'Radon',
                'Fr': 'Francium',
                'Ra': 'Radium',
                'Ac': 'Actinium',
                'Th': 'Thorium',
                'Pa': 'Protactinium',
                'U': 'Uranium',
                'Np': 'Neptunium',
                'Pu': 'Plutonium',
                'Am': 'Americium',
                'Cm': 'Curium',
                'Bk': 'Berkelium',
                'Cf': 'Californium',
                'Es': 'Einsteinium',
                'Fm': 'Fermium',
                'Md': 'Mendelevium',
                'No': 'Nobelium',
                'Lr': 'Lawrencium',
                'Rf': 'Rutherfordium',
                'Db': 'Dubnium',
                'Sg': 'Seaborgium',
                'Bh': 'Bohrium',
                'Hs': 'Hassium',
                'Mt': 'Meitnerium',
                'Ds': 'Darmstadtium',
                'Rg': 'Roentgenium',
                'Cn': 'Copernicium',
                'Nh': 'Nihonium',
                'Fl': 'Flerovium',
                'Mc': 'Moscovium',
                'Lv': 'Livermorium',
                'Ts': 'Tennessine',
                'Og': 'Oganesson'}


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
        text += f'{count} {periodic_table[elem]}, '
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
    
    def convert(self, db_name, split):
        '''
        Convert figshare data to json
        '''
        # open figshare data and convert to text and graph
        data = pd.DataFrame(jdata(db_name))
        if split >= 0:
            data = np.array_split(data, 4)[split]
        # split series using np
        data = np.array_split(data, 12)

        results = Parallel(n_jobs=12)(delayed(self.treat_chunk)(chunk) for chunk in tqdm(data))
        results = pd.DataFrame(sum(results, []))
        # results = pd.concat(results)
        # results = results.to_list()
        # save to json
        if split >= 0:
            results.to_parquet(f'{db_name}_{split+1}_data.parquet')
            # with open(f'{db_name}_{split+1}_data.json', 'w') as f:
            #     json.dump(results, f)
        else:
            results.to_parquet(f'{db_name}_data.parquet')
            # with open(f'{db_name}_data.json', 'w') as f:
            #     json.dump(results, f)
        return results
    
    def treat_chunk(self, chunk):
        result = []
        for _, row in chunk.iterrows():
            try:
                item = self.generate_graph_and_text(row['atoms'])
                # for k, v in item.items():
                #     row[k] = v
                result.append(item)
            except:
                pass
        # result = pd.Series(result)
        return result

    def generate_graph_and_text(self, atoms):
        '''
        Generate graph and text for a single atomic structure
            atoms: dict
            items: dict
        '''
        jatoms = Atoms.from_dict(atoms)
        # Generate graph
        dglgraph = Graph.atom_dgl_multigraph(jatoms, compute_line_graph=False, atom_features='cgcnn')
        item = {}
        edges = dglgraph.edges()
        item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
        item['num_nodes'] = dglgraph.num_nodes()
        item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
        item['edge_attr'] = dglgraph.edata['r'].tolist()
        item['atoms'] = atoms
        
        # Generate text
        formula = jatoms.composition.reduced_formula
        crystalsystem = Spacegroup3D(jatoms).crystal_system
        count_elems_text = print_element_counts(count_elements(jatoms.elements))

        item['y'] = f'A POSCAR of the {crystalsystem} {formula}. \n{count_elems_text} And it has {crystalsystem} structure.'
        return item
    

def main():
    dbs = ['dft_3d_2021']
    split = -1
    converter = JarvisToJson()
    for db in dbs:
        converter.convert(db, split)


if __name__ == '__main__':
    main()