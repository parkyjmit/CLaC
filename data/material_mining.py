'''
From JARVIS figshare data, generate cif files and find paragraphs in materials science papers
generate_cif: if True, generate cif files from JARVIS figshare data
find_paragraphs: if True, find paragraphs in materials science papers
'''
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
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

def search_paper_paragraphs(df, mat):
    # First check if the formula is in the text
    df['query'] = df['text'].apply(lambda x: mat in x)
    true_rows = df[df['query']]
    selected_rows = true_rows if len(true_rows) <= 2 else true_rows.sample(n=2) # max 2
    text_series = selected_rows['text'].apply(lambda x: [p for p in x.split('\n') if mat in p][:2]).to_list()  # list of paragraph including keyword
    doi_series = selected_rows['doi'].to_list()
    return text_series, doi_series

def cif_generate_chunk(chunk):
        result = []
        for atom in tqdm(chunk['atoms']):
            try:
                jatoms = Atoms.from_dict(atom)
                item = jatoms.generate_cif_string()
                item['atoms'] = atom
                result.append(item)
            except: pass
        return result


def main(args):
    # Load all materials db and deduplicate
    if args['generate_cif']:
        dbs = args['db_name']
        atoms_volumes = []
        for db in dbs:
            db = pd.DataFrame(jdata(db))
            db_split = np.array_split(db, 4)
            for i in range(4):
                chunks = db_split[i]
                chunks = np.array_split(chunks, 12)
                results = Parallel(n_jobs=12)(delayed(cif_generate_chunk)(chunk) for chunk in chunks)
                results =  sum(results, [])
                atoms_volumes += results    
        atoms_volumes = pd.DataFrame(atoms_volumes)
        # atoms_volumes['drop_key'] = atoms_volumes['reduced_formula'] + atoms_volumes['crystal system'] + atoms_volumes['spg']
        # atoms_volumes = atoms_volumes.sort_values('volume').drop_duplicates(subset=['drop_key'], keep='first')
        # atoms_volumes = atoms_volumes.drop(columns=['drop_key'])
        # Save as parquet
        atoms_volumes.to_parquet(args['output']+'.parquet')
        del db

    if args['find_paragraphs']:
        atoms_volumes = pd.read_parquet(args['output']+'.parquet')  # load file contarining cif data
        chunks = np.array_split(atoms_volumes, 12)

        def search_paragraphs_chunk(chunk):
            '''
            search paragraphs from materials database chunk
            '''
            get_result = []
            for mat in tqdm(chunk.iloc):
                query = mat['reduced_formula']
                if len(query) <= 2:
                    if len(query) == 1:
                        query = periodic_table[query]
                    else:
                        if query[1].islower():
                            query = periodic_table[query]
                        else:
                            query = query
                result = search_paper_paragraphs(df, mat['reduced_formula'])
                mat['paragraphs'] = result[0]
                mat['doi'] = result[1]
                mat['num_papers'] = len(result[1])
                if mat['num_papers'] > 0:
                    get_result.append(mat)
            return get_result

        # Repeat for materials science papers 10 times
        for i in range(10):
            # Load paper data
            df = pq.read_table(f'/mnt/hdd1/SESPapers/multi-label/materials_paper_split_{i}.parquet')
            df = df.to_pandas()
            results = Parallel(n_jobs=12)(delayed(search_paragraphs_chunk)(chunk) for chunk in chunks)
            results = pd.DataFrame(sum(results, []))

            # Save as parquet
            results.to_parquet(args['output_materials']+f'_split_{i}.parquet')
            del results
        

if __name__ == '__main__':
    args = {}
    args['generate_cif'] = False
    args['find_paragraphs'] = True
    args['db_name'] = ['aflow2']
    args['materials_science_papers_1'] = '/mnt/hdd1/SESPapers/multi-label/paper_type7.parquet'
    args['materials_science_papers_2'] = '/mnt/hdd1/SESPapers/multi-label/paper_type2.parquet'
    args['output'] = 'aflow2_cif'
    args['output_materials'] = 'aflow2_cif_papers'

    main(args)