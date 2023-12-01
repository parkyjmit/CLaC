'''
From text file containing list of liquid electrolytes, search for the paragraphs in the materials science papers
find_paragraphs: if True, search for paragraphs in the battery papers
'''
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
from joblib import Parallel, delayed

from utils import pubchem_request, search_mat_in_paper_return_paragraphs


def main(args):
    if args['find_paragraphs']:
        # Load liquid electrolyte data and prepare chunking
        electrolytes = pd.read_json(args['list_of_electrolytes'])
        # with open(args['list_of_electrolytes'], 'r') as f:
        #     electrolytes = f.readlines()
        # electrolytes = [elect.split('\n')[0] for elect in electrolytes]
        elec_chunks = np.array_split(electrolytes, 12)

        # Parallel processing
        def treat_chunk(chunk):
            '''
            input: list of str
            item: dict
                SMILES, InChI, name, list(papers(list(paragraphs))), num_papers
            return: list of dict
            '''
            get_result = [] 
            for _, elect in tqdm(chunk.iterrows()):
                synonyms = elect['Synonym']
                item = {k: v for k, v in elect.items()}
                item['paragraphs'] = []
                item['doi'] = []
                item['num_papers'] = 0
                for syn in synonyms[:10]:
                    result = search_mat_in_paper_return_paragraphs(df, syn)
                    item['paragraphs'] += result[0]
                    item['doi'] += result[1]
                    item['num_papers'] += len(result[1])
                if item['num_papers'] > 0:
                    get_result.append(item)
            return get_result
        
        for i in range(10):
            if args['search_only_in_battery_papers']:
                # load battery papers
                df = pd.read_json(args['battery_papers'])
            else:
                # load materials science papers
                df = pd.read_parquet((f'/mnt/hdd1/SESPapers/multi-label/materials_paper_split_{i}.parquet'))
            # treat_chunk(elec_chunks[0])
            parallel_result = Parallel(n_jobs=12)(delayed(treat_chunk)(chunk) for chunk in elec_chunks)
            parallel_result = pd.DataFrame(sum(parallel_result, []))
            
            # Save as parquet
            if args['search_only_in_battery_papers']:
                parallel_result.to_parquet(args['output_materials']+f'_from_battery_paper.parquet')
                break
            else:
                parallel_result.to_parquet(args['output_materials']+f'_split_{i}.parquet')

    if args['find_similar_molecules']:
        # Load liquid electrolyte data and prepare chunking
        with open(args['list_of_electrolytes'], 'r') as f:
            electrolytes = f.readlines()
        electrolytes = [elect.split('\n')[0] for elect in electrolytes]
        pubchem_request(elect['CanonicalSMILES'], key='smiles', task='fastsimilarity_2d', max_records=100)
    # with open(args['output_battery'], 'r') as f:
    #     elect_with_paragraphs = json.load(f)
    # # expand papers from battery to materials science
    # table = pq.read_table(args['materials_science_papers_1'])
    # df_1 = table.to_pandas()
    # table = pq.read_table(args['materials_science_papers_2'])
    # df_2 = table.to_pandas()
    # df = pd.concat([df_1, df_2])
    # del df_1, df_2
    
    # electrolytes = [elect for elect in elect_with_paragraphs]  # change to smiles
    # elec_chunks = np.array_split(electrolytes, 1)
    # # Parallel processing
    # def treat_chunk_with_similarity(chunk):
    #     '''
    #     input: list of str
    #     items: list of dict
    #     item: dict
    #         SMILES, InChI, name, list(papers(list(paragraphs))), num_papers
    #     return: list of dict
    #     '''
    #     get_result = [] 
    #     for elect in tqdm(chunk):
    #         try:
    #             # find similar molecules with anchor
    #             items = pubchem_request(elect['CanonicalSMILES'], key='smiles', task='fastsimilarity_2d', max_records=100)
    #             print(len(items))
    #             for item in items:
    #                 try:  # If not found, skip
    #                     result = search_mat_in_paper_return_paragraphs(df, item['IUPACName'])
    #                     item['paragraphs'] = result[0]
    #                     item['doi'] = result[1]
    #                     item['num_papers'] = len(result[1])
    #                     if item['num_papers'] > 0:
    #                         get_result.append(item)
    #                 except: pass# Exception as e:
    #                     # print('error', e)
    #         except: pass# Exception as e:
    #             # print('itemerror', e)
    #         print(len(get_result))
    #     return get_result
    # # parallel_result = Parallel(n_jobs=2)(delayed(treat_chunk_with_similarity)(chunk) for chunk in elec_chunks)
    # parallel_result = treat_chunk_with_similarity(elec_chunks[0])

    # # # remove duplicates
    # # battery_json = pd.DataFrame(elect_with_paragraphs)
    # # similarity_json = pd.DataFrame(elect_with_paragraphs_sim)

    # # similarity_json = similarity_json[~similarity_json['CID'].isin(battery_json['CID'])]
    # # to_json = []
    # # for item in similarity_json.iloc:
    # #     to_json.append({k: v for k, v in item.items()})

    # # Save as json
    # with open(args['output_similarity'], 'w') as f:
    #     # json.dump(to_json, f)
    #     json.dump(parallel_result, f)

    


if __name__ == '__main__':
    args = {}
    args['find_paragraphs'] = True
    args['find_similar_molecules'] = False
    args['search_only_in_battery_papers'] = False
    args['list_of_electrolytes'] = '/home/yj/PycharmProjects/MIT/CLaMP/jsons/basic_mol_list.json'
    args['battery_papers'] = '/mnt/hdd1/SESPapers/ses_papers_battery_reclassification.json'
    # args['materials_science_papers_1'] = '/mnt/hdd1/SESPapers/multi-label/paper_type7.parquet'
    # args['materials_science_papers_2'] = '/mnt/hdd1/SESPapers/multi-label/paper_type2.parquet'
    # args['output_battery'] = 'liquid_electrolytes_data.json'
    # args['output_similarity'] = 'liquid_electrolytes_with_similarity_data.json'
    args['output_materials'] = 'real_basic_mols_with_paragraphs'
    main(args)