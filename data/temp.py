import pandas as pd


def classifier(x):
    if x == 'orthorhombic':
        return [[1, 0, 0, 0, 0, 0, 0]]
    elif x == 'cubic':
        return [[0, 1, 0, 0, 0, 0, 0]]
    elif x == 'tetragonal':
        return [[0, 0, 1, 0, 0, 0, 0]]
    elif x == 'trigonal':
        return [[0, 0, 0, 1, 0, 0, 0]]
    elif x == 'monoclinic':
        return [[0, 0, 0, 0, 1, 0, 0]]
    elif x == 'triclinic':
        return [[0, 0, 0, 0, 0, 1, 0]]
    elif x == 'hexagonal':
        return [[0, 0, 0, 0, 0, 0, 1]]

for dataset in ['mp_3d_2020_materials_graphs_gpt_questions', 'mp_3d_2020_merged']:
    for split in ['train', 'val', 'test']:
        name = f'/home/lucky/Projects/CLaMP/datafiles/{dataset}_{split}.parquet'
        df = pd.read_parquet(name)
        df['y'] = df['crystal system'].apply(lambda x: classifier(x))
        df.to_parquet(name)