'''
Generate cif files
'''
from tqdm import tqdm
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import argparse
from utils import periodic_table, convert_arrays_to_lists


def cif_generate_chunk(chunk):
    '''
    Generate cif files as text.
    args:
        chunk: chunk of data
    return:
        result: list of cif files
    '''    
    result = []
    for i, row in tqdm(chunk.iterrows()):
        try:
            jatoms = Atoms.from_dict(convert_arrays_to_lists(row['atoms']))
            pymat = jatoms.pymatgen_converter()
            cif = pymat.to(fmt='cif')
            # copy of row
            item = row.copy()
            item['cif'] = cif
            result.append(item)
        except: pass
    return result


def main(args):
    # Load all materials db and deduplicate
    if args.generate_cif:
        n_jobs = 64
        atoms_volumes = []
        db = pd.read_parquet(hf_hub_download(repo_id='yjeong/GPT-Narratives-for-Materials', filename=args.input_filename, repo_type="dataset"))
        db_split = np.array_split(db, 4)
        for i in range(4):
            chunks = db_split[i]
            chunks = np.array_split(chunks, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(cif_generate_chunk)(chunk) for chunk in chunks)
            results =  sum(results, [])
            atoms_volumes += results    
        atoms_volumes = pd.DataFrame(atoms_volumes)
        # Save as parquet
        atoms_volumes.to_parquet(args.filename)
        del db

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cif files')
    parser.add_argument('--filename', type=str, default='mp_3d_2020_gpt_narratives.parquet', help='materials file name')
    args = parser.parse_args()

    main(args)