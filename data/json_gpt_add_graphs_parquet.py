from utils import json_atoms_to_graphs
import pandas as pd
import argparse


def main(args):
    '''
    Convert atoms in json data to graph and combine with original data
    Once saved as parquet file, 'atoms' column cannot be read. Therefore, data are saved as json first.
    input: json data path
        atoms, properties, text
    output: parquet data path
        atoms, properties, text, node_feat, edge_index, edge_attr, num_nodes
    '''
    df = pd.read_json(args.data_path)
    results = json_atoms_to_graphs(args.data_path)
    df_results = pd.DataFrame(results)
    new_df = pd.concat([df, df_results], axis=1)
    # new_df['y'] = new_df['atoms'].apply(lambda x: [0])
    new_df.to_parquet(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials.json')
    parser.add_argument('--output-path', type=str, default='/home/yj/PycharmProjects/MIT/CLaMP/jsons/mp_3d_2020_materials_graphs_gpt.parquet')
    args = parser.parse_args()
    main(args)