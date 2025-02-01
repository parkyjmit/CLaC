'''
train, val, test split여기서
미리 edge_index 등 그래프 구성도 여기서
jarvis figshare data에서 graph-text pair를 여기서 만들어서 json으로 저장
'''
from tqdm import tqdm
import pandas as pd
import numpy as np
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
from joblib import Parallel, delayed
from utils import convert_arrays_to_lists
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
import torch
from rdkit.Chem.rdchem import BondType as BT
import argparse


def main(args):
    df = pd.read_parquet(os.path.join(cur_dir, args.output+'.parquet'))

    def process_row(row):
        item = row.copy()
        jatoms = Atoms.from_dict(convert_arrays_to_lists(row['atoms']))
        # Generate graph
        dglgraph = Graph.atom_dgl_multigraph(jatoms, compute_line_graph=False, atom_features='cgcnn')
        edges = dglgraph.edges()
        item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
        item['num_nodes'] = dglgraph.num_nodes()
        item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
        item['edge_attr'] = dglgraph.edata['r'].tolist()
        item['y'] = [0]
        return item

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in tqdm(df.iterrows()))

    df = pd.DataFrame(results)

    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.iloc[:int(len(df)*0.9)]
    val_df = df.iloc[int(len(df)*0.9):int(len(df)*0.95)]
    test_df = df.iloc[int(len(df)*0.95):]
    print(len(train_df), len(val_df), len(test_df))
    train_df.to_parquet(os.path.join(cur_dir, args.output+'_train.parquet'))
    val_df.to_parquet(os.path.join(cur_dir, args.output+'_val.parquet'))
    test_df.to_parquet(os.path.join(cur_dir, args.output+'_test.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graph')
    parser.add_argument('--output', type=str, default='mp_3d_2020_gpt_narratives', help='')
    args = parser.parse_args()

    main(args)