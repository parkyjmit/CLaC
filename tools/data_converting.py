
import torch
from torch_geometric.data import Data
import numpy as np
from datasets import load_dataset
from ase import Atoms
from ase.geometry import cellpar_to_cell
import numpy as np

def dict_to_atoms(d: dict) -> Atoms:
    # 1) 셀(cell) 만들기: lattice_mat이 있으면 그대로, 없으면 abc+angles로 변환
    if 'lattice_mat' in d and d['lattice_mat'] is not None:
        cell = d['lattice_mat']  # 3x3 matrix (Å)
    else:
        # abc(Å) + angles(°) -> 6개 파라미터를 cell 행렬로
        abc = d['abc']          # [a, b, c]
        angles = d['angles']    # [alpha, beta, gamma]
        cell = cellpar_to_cell(abc + angles)  # Å, °

    # 2) 원소 목록
    symbols = d['elements']

    # 3) 좌표: cartesian이면 positions, 아니면 scaled_positions(분수좌표)
    if d.get('cartesian', False):
        atoms = Atoms(symbols=symbols,
                      positions=d['coords'],
                      cell=cell,
                      pbc=True)
    else:
        atoms = Atoms(symbols=symbols,
                      scaled_positions=d['coords'],
                      cell=cell,
                      pbc=True)

    # 4) 부가 정보가 있으면 info에 보관(선택)
    if 'props' in d:
        atoms.info['props'] = d['props']

    return atoms
# 필요한 패키지
import numpy as np
import torch
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list

def ase_to_pyg(
    atoms,
    cutoff: float = 6.0,
    add_vectors: bool = True,
):
    """
    ASE Atoms -> torch_geometric.data.Data
    PBC 고려: neighbor_list('ijS')로 셀 쉬프트 S를 받아 edge 벡터/거리 계산

    Args:
        atoms: ase.Atoms
        cutoff: 이웃 탐색 컷오프(Å)
        add_vectors: True면 edge 벡터와 길이 추가

    Returns:
        Data(
          z, pos, cell, pbc,
          edge_index, (선택)edge_vec, (선택)edge_length,
          S, cell_offsets
        )
    """

    # --- 노드(원자) 정보 ---
    pos = atoms.get_positions()                          # (N,3) cartesian Å
    z = atoms.get_atomic_numbers()                       # (N,)
    cell = atoms.cell.array.astype(np.float64)           # (3,3)
    pbc = np.asarray(atoms.pbc, dtype=bool)              # (3,)

    # --- PBC 이웃 탐색 ---
    # 'ijS' : i(중심), j(이웃), S(정수 셀 쉬프트). S @ cell 이 주기 이미지의 이동 벡터(Å).
    i_idx, j_idx, S = neighbor_list('ijS', atoms, cutoff)

    # --- edge 기하 계산(최소이미지) ---
    # displacement: r_j + S @ cell - r_i
    # S 는 (E,3) 정수, cell은 (3,3) 이므로, (S @ cell) -> (E,3)
    if S.size == 0:
        # 고립 원자 등 이웃이 없을 수 있음
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            cell=torch.tensor(cell, dtype=torch.float32),
            pbc=torch.tensor(pbc, dtype=torch.bool),
            edge_index=edge_index,
            num_nodes=len(z),
        )
        return data

    cell_offsets = S @ cell  # (E,3) in Å
    disp = pos[j_idx] + cell_offsets - pos[i_idx]  # (E,3)

    # 2) 노드 특성
    atomic_nums = atoms.get_atomic_numbers()
    atom_features = [get_atom_fea(z) for z in atomic_nums]
    x = torch.tensor(np.vstack(atom_features), dtype=torch.float32)

    # --- PyG 텐서화 ---
    edge_index = torch.tensor(
        np.vstack([i_idx, j_idx]), dtype=torch.long
    )
    data_kwargs = dict(
        x=x,  # (N, F) 원자 특성 벡터
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        cell=torch.tensor(cell, dtype=torch.float32),
        pbc=torch.tensor(pbc, dtype=torch.bool),
        edge_index=edge_index,
        # 저장해두면 재구성/시각화/특징 생성에 유용
        S=torch.tensor(S, dtype=torch.int64),  # 정수 셀 쉬프트
        cell_offsets=torch.tensor(cell_offsets, dtype=torch.float32),  # Å
        num_nodes=len(z),
    )

    if add_vectors:
        edge_vec = torch.tensor(disp, dtype=torch.float32)                 # (E,3)
        # edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)        # (E,1)
        # 관습적으로 edge_attr에 스칼라 거리만 두거나, 벡터를 별도로 둡니다.
        data_kwargs.update(
            edge_attr=edge_vec,
            # edge_length=edge_len,               # 모델에 따라 edge_attr로 써도 됨
            # edge_attr=edge_len,               # 이렇게 쓰면 바로 Edge Attr로 활용 가능
        )

    data = Data(**data_kwargs)
    return data


def convert_arrays_to_lists(data):
    if isinstance(data, dict):
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist() if data.dtype != object else [convert_arrays_to_lists(item) for item in data]
    else:
        return data

def get_atoms_from_file(file_path):
    """
    Loads the first 'atoms' entry from a parquet dataset file.
    """
    print(f"Loading dataset from {file_path}...")
    try:
        dataset = load_dataset('parquet', data_files=file_path, split='train')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    first_item = dataset[0]
    atoms_dict = first_item['atoms']
    atoms_dict_lists = convert_arrays_to_lists(atoms_dict)
    return dict_to_atoms(atoms_dict_lists)

def atom_to_torch_graph_data_with_dgl(jatoms):
    """
    Converts a jarvis.core.atoms.Atoms object to a torch_geometric.data.Data object
    by first creating a DGL graph. This is the reference implementation.
    """
    dglgraph = Graph.atom_dgl_multigraph(
        jatoms,
        cutoff=0.0,
        compute_line_graph=False,
        atom_features='cgcnn'
    )
    src, dst = dglgraph.edges()
    x = dglgraph.ndata['atom_features']
    edge_attr = dglgraph.edata['r']
    
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.stack([src, dst], dim=0),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.zeros(1, dtype=torch.float32),
    )

import numpy as np
import torch
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list
from typing import Any, Dict, List, Mapping, Optional

def ase_to_pyg_knn(
    atoms,
    cutoff: float = 6.0,
    add_vectors: bool = True,
    k: Optional[int] = None,
    cutoff_start: float = 3.0,
    cutoff_step: float = 0.75,
    cutoff_max: float = 12.0,
    eps: float = 1e-8,
):
    """
    ASE Atoms -> torch_geometric.data.Data
    PBC 고려: neighbor_list('ijS')로 셀 쉬프트 S를 받아 edge 벡터/거리 계산

    Args:
        atoms: ase.Atoms
        cutoff: 이웃 탐색 컷오프(Å). k가 None일 때 사용.
        add_vectors: True면 edge 벡터와 길이 추가
        k: 각 원자당 최근접 이웃 수. 지정되면 adaptive cutoff를 사용해 k명을 확보.
        cutoff_start/cutoff_step/cutoff_max: k-NN 탐색 시 사용할 초기 컷오프, 증가 폭, 최대 컷오프.
        eps: 자기 자신 등 0거리 엣지를 제거하기 위한 허용 오차

    Returns:
        Data(
          z, pos, cell, pbc,
          edge_index, (선택)edge_vec, (선택)edge_length,
          S, cell_offsets
        )
    """
    atoms = dict_to_atoms(atoms) if isinstance(atoms, dict) else atoms
    # --- 노드(원자) 정보 ---
    pos = atoms.get_positions()                          # (N,3) cartesian Å
    z = atoms.get_atomic_numbers()                       # (N,)
    cell = atoms.cell.array.astype(np.float64)           # (3,3)
    pbc = np.asarray(atoms.pbc, dtype=bool)              # (3,)

    atom_features = [get_atom_fea(int(z_val)) for z_val in z]
    x = torch.tensor(np.vstack(atom_features), dtype=torch.float32)

    def _empty_graph() -> Data:
        data_kwargs = dict(
            x=x,
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            cell=torch.tensor(cell, dtype=torch.float32),
            pbc=torch.tensor(pbc, dtype=torch.bool),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            S=torch.zeros((0, 3), dtype=torch.int64),
            cell_offsets=torch.zeros((0, 3), dtype=torch.float32),
            num_nodes=len(z),
        )
        if add_vectors:
            data_kwargs['edge_attr'] = torch.zeros((0, 3), dtype=torch.float32)
            data_kwargs['edge_length'] = torch.zeros((0, 1), dtype=torch.float32)
        return Data(**data_kwargs)

    if k is None:
        # --- PBC 이웃 탐색 ---
        # 'ijS' : i(중심), j(이웃), S(정수 셀 쉬프트). S @ cell 이 주기 이미지의 이동 벡터(Å).
        i_idx, j_idx, S = neighbor_list('ijS', atoms, cutoff)

        # --- edge 기하 계산(최소이미지) ---
        # displacement: r_j + S @ cell - r_i
        # S 는 (E,3) 정수, cell은 (3,3) 이므로, (S @ cell) -> (E,3)
        if S.size == 0:
            return _empty_graph()

        cell_offsets = S @ cell  # (E,3) in Å
        disp = pos[j_idx] + cell_offsets - pos[i_idx]  # (E,3)

        # --- PyG 텐서화 ---
        edge_index = torch.tensor(
            np.vstack([i_idx, j_idx]), dtype=torch.long
        )
        data_kwargs = dict(
            x=x,  # (N, F) 원자 특성 벡터
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            cell=torch.tensor(cell, dtype=torch.float32),
            pbc=torch.tensor(pbc, dtype=torch.bool),
            edge_index=edge_index,
            # 저장해두면 재구성/시각화/특징 생성에 유용
            S=torch.tensor(S, dtype=torch.int64),  # 정수 셀 쉬프트
            cell_offsets=torch.tensor(cell_offsets, dtype=torch.float32),  # Å
            num_nodes=len(z),
        )

        if add_vectors:
            edge_vec = torch.tensor(disp, dtype=torch.float32)                 # (E,3)
            edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
            data_kwargs.update(
                edge_attr=edge_vec,
                edge_length=edge_len,
            )

        return Data(**data_kwargs)

    # k-NN 그래프 구성
    N = len(z)
    neigh_idx = [[] for _ in range(N)]
    neigh_S: List[List[List[int]]] = [[] for _ in range(N)]
    neigh_disp = [[] for _ in range(N)]
    neigh_dist = [[] for _ in range(N)]

    cutoff_cur = float(cutoff_start)

    def need_more() -> bool:
        return any(len(neigh_idx[i]) < k for i in range(N))

    while need_more() and cutoff_cur <= cutoff_max:
        ii, jj, S = neighbor_list('ijS', atoms, cutoff_cur)
        if ii.size == 0:
            cutoff_cur += cutoff_step
            continue

        cell_offsets = S @ cell
        disp_all = pos[jj] + cell_offsets - pos[ii]
        dist_all = np.linalg.norm(disp_all, axis=1)

        valid = dist_all > eps
        ii, jj, S = ii[valid], jj[valid], S[valid]
        disp_all, dist_all = disp_all[valid], dist_all[valid]

        for i in range(N):
            if len(neigh_idx[i]) >= k:
                continue
            mask_i = (ii == i)
            if not np.any(mask_i):
                continue

            cand_j = jj[mask_i]
            cand_S = S[mask_i]
            cand_disp = disp_all[mask_i]
            cand_dist = dist_all[mask_i]

            if neigh_idx[i]:
                seen = set(zip(neigh_idx[i], map(tuple, neigh_S[i])))
                sel_mask = np.array([
                    (int(j), tuple(s)) not in seen for j, s in zip(cand_j, cand_S)
                ])
                cand_j, cand_S = cand_j[sel_mask], cand_S[sel_mask]
                cand_disp, cand_dist = cand_disp[sel_mask], cand_dist[sel_mask]

            if cand_j.size == 0:
                continue

            order = np.argsort(cand_dist)
            take = int(min(k - len(neigh_idx[i]), cand_j.size))
            order = order[:take]

            neigh_idx[i].extend(list(map(int, cand_j[order])))
            neigh_S[i].extend([list(map(int, s)) for s in cand_S[order]])
            neigh_disp[i].extend([disp.tolist() for disp in cand_disp[order]])
            neigh_dist[i].extend([float(dist) for dist in cand_dist[order]])

        cutoff_cur += cutoff_step

    rows, cols, S_list, disp_list = [], [], [], []
    for i in range(N):
        m = len(neigh_idx[i])
        if m == 0:
            continue
        rows.extend([i] * m)
        cols.extend(neigh_idx[i])
        S_list.extend(neigh_S[i])
        disp_list.extend(neigh_disp[i])

    if not rows:
        return _empty_graph()

    S_arr = np.asarray(S_list, dtype=np.int64)
    cell_offsets = S_arr @ cell
    edge_vec_np = np.asarray(disp_list, dtype=np.float32)
    edge_len_np = np.linalg.norm(edge_vec_np, axis=1, keepdims=True)

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    data_kwargs = dict(
        x=x,
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        cell=torch.tensor(cell, dtype=torch.float32),
        pbc=torch.tensor(pbc, dtype=torch.bool),
        edge_index=edge_index,
        S=torch.tensor(S_arr, dtype=torch.int64),
        cell_offsets=torch.tensor(cell_offsets, dtype=torch.float32),
        num_nodes=N,
    )
    if add_vectors:
        data_kwargs.update(
            edge_attr=torch.tensor(edge_vec_np, dtype=torch.float32),
            edge_length=torch.tensor(edge_len_np, dtype=torch.float32),
        )

    return Data(**data_kwargs)




import torch
from torch_geometric.data import Data
# from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
# AtomCustomJSONInitializer 는 기존 그대로 사용한다고 가정
import json

def get_atom_fea(z, atom_init_path='data/atom_init.json'):
    """
    원자 번호 z에 대해 CGCNN 스타일의 원자 특성 벡터를 반환합니다.
    """
    featurizer = json.load(open(atom_init_path))
    return featurizer[str(z)]

def atom_to_torch_graph_data_direct(jatoms, cutoff=0.0, atom_init_path='data/atom_init.json'):
    """
    jarvis.core.atoms.Atoms -> torch_geometric.data.Data
    Edge는 JARVIS의 neighbor 유틸로 구성.
    """
    
    # 1) 엣지와 변위 벡터(r) 만들기
    edges = nearest_neighbor_edges(atoms=jatoms, cutoff=cutoff, max_neighbors=12, use_canonize=True)
    # edges: dict[(src_id, dst_id)] -> set(dst_image)
    # 아래가 (u, v, r) 텐서를 돌려줍니다.
    u, v, r = build_undirected_edgedata(atoms=jatoms, edges=edges)  # r: (E,3) cartesian displacement
    if u.numel() == 0:
        raise ValueError("생성된 엣지가 없습니다. cutoff를 키우거나 구조를 확인하세요.")

    edge_index = torch.stack([u.long(), v.long()], dim=0)      # (2, E)
    edge_attr  = r.to(torch.float32)                           # (E, 3) 벡터 특성
    # 필요하면 거리 스칼라도 추가 가능:
    # dist = torch.linalg.norm(edge_attr, dim=1, keepdim=True)  # (E,1)

    # 2) 노드 특성
    atomic_nums = jatoms.atomic_numbers
    atom_features = [get_atom_fea(z) for z in atomic_nums]
    x = torch.tensor(np.vstack(atom_features), dtype=torch.float32)

    # 3) PyG Data
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,  # 또는 torch.cat([edge_attr, dist], dim=1)
        y=torch.zeros(1, dtype=torch.float32),
    )

def main():
    """
    Main function to run the experiment and compare the two methods.
    """
    # Define paths
    # NOTE: The user mentioned this file path is a copy.
    data_file_path = '/home/lucky/Projects/CLaC copy/datafiles/mp_3d_2020_gpt_narratives_train.parquet'
    atom_init_path = '/home/lucky/Projects/CLaC-revision/atom_init.json'

    # 1. Get Atoms object from the specified file
    jatoms = get_atoms_from_file(data_file_path)
    
    if jatoms is None:
        print("Could not proceed with the conversion experiment.")
        return

    print("Successfully created ase.Atoms object from file.")
    print(jatoms)
    print("-" * 40)

    # # 2. Convert using the DGL-based method
    # print("Converting with DGL-based method...")
    # data_dgl = atom_to_torch_graph_data_with_dgl(jatoms)
    # print("Result from DGL-based method:")
    # print(data_dgl)
    # print("-" * 40)

    # 3. Convert using the direct method
    print("Converting with direct method (no DGL)...")
    data_direct = ase_to_pyg(jatoms, cutoff=6.0)
    print("Result from direct method:")
    print(data_direct)
    print("-" * 40)

    print("Converting with direct method (no DGL)...")
    data_direct = ase_to_pyg_knn(jatoms, cutoff_max=8.0, k=12)
    print("Result from direct method:")
    print(data_direct)

    # 4. Compare the results
    print("Comparing the results...")
    
    # # Compare node features
    # x_match = torch.allclose(data_dgl.x, data_direct.x)
    # print(f"Node features (x) match: {x_match}")

    # # Compare edge indices
    # num_nodes = data_dgl.num_nodes
    # if num_nodes != data_direct.num_nodes:
    #     edge_index_match = False
    # else:
    #     dgl_edges_canonical = data_dgl.edge_index[0] * num_nodes + data_dgl.edge_index[1]
    #     dgl_edges_sorted, dgl_perm = torch.sort(dgl_edges_canonical)

    #     direct_edges_canonical = data_direct.edge_index[0] * num_nodes + data_direct.edge_index[1]
    #     direct_edges_sorted, direct_perm = torch.sort(direct_edges_canonical)

    #     edge_index_match = torch.equal(dgl_edges_sorted, direct_edges_sorted)
    
    # print(f"Edge indices (edge_index) match: {edge_index_match}")

    # # Compare edge attributes
    # edge_attr_match = False
    # if edge_index_match and data_dgl.edge_attr.shape == data_direct.edge_attr.shape:
    #     edge_attr_dgl_sorted = data_dgl.edge_attr.flatten()[dgl_perm]
    #     edge_attr_direct_sorted = data_direct.edge_attr.flatten()[direct_perm]
    #     edge_attr_match = torch.allclose(edge_attr_dgl_sorted, edge_attr_direct_sorted)
    #     print(f"Edge attributes (edge_attr) match: {edge_attr_match}")
    # else:
    #     print("Edge attributes (edge_attr) not compared due to shape or index mismatch.")

    # print("-" * 40)
    # if x_match and edge_index_match and edge_attr_match:
    #     print("Conclusion: The direct conversion method produces an equivalent PyG Data object.")
    # else:
    #     print("Conclusion: The two methods produce different results.")


if __name__ == "__main__":
    main()
