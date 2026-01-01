from typing import Any, Dict, List, Mapping, Optional
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from model.text_encoder import CLaCTokenizer, decoder_model_list
from data.utils import convert_arrays_to_lists
from data.augmentation import perturb_structure, apply_strain
from orb_models.forcefield import pretrained, atomic_system
from orb_models.forcefield.base import batch_graphs
from transformers.data.data_collator import default_data_collator
# from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
# from data.utils import jarvis_atoms2graph
# AtomCustomJSONInitializer 는 기존 그대로 사용한다고 가정
import json
import numpy as np
from ase.neighborlist import neighbor_list
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read
from io import StringIO
from torch_geometric.data import Data, Batch
import random
import re
import spacy
nlp = spacy.load('en_core_web_sm')


def replace_chemical_formula(text, replacement="material"):
    """
    Replace chemical formulas in text with a generic word.

    Handles formulas like:
    - Zn(AgO2)2, Ca3V3(AsO4)4, NaSr(PO3)3 (with parentheses)
    - BiO2, Te2MoWSeS (simple formulas with multiple elements)

    Args:
        text: Input text containing chemical formulas
        replacement: Word to replace formulas with (default: "material")

    Returns:
        Text with chemical formulas replaced
    """
    # Pattern for formulas with parentheses: e.g., Zn(AgO2)2, Ca3V3(AsO4)4
    formula_with_paren = r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\)\d+\b'

    # Pattern for simple formulas (at least 2 elements): e.g., BiO2, NaCl
    formula_simple = r'\b(?:[A-Z][a-z]?\d*){2,}\b'

    # First, replace formulas with parentheses (more specific)
    text = re.sub(formula_with_paren, replacement, text)

    # Then replace simple formulas (exclude short ones like C2, or IDs)
    def is_valid_formula(match):
        formula = match.group(0)
        # Exclude pure digits
        if formula.isdigit():
            return False
        # Exclude very short formulas (2 chars or less)
        if len(formula) <= 2:
            return False
        return True

    text = re.sub(formula_simple, lambda m: replacement if is_valid_formula(m) else m.group(0), text)

    return text


def get_atom_fea(z, atom_init_path='data/atom_init.json'):
    """
    원자 번호 z에 대해 CGCNN 스타일의 원자 특성 벡터를 반환합니다.
    """
    featurizer = json.load(open(atom_init_path))
    return featurizer[str(z)]


def cif_to_atoms(cif_string: str) -> Atoms:
    """
    CIF 문자열을 ASE Atoms 객체로 변환합니다.
    """
    try:
        import warnings
        # CIF 파싱 시 space group 경고 무시
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="ase.spacegroup")
            # StringIO를 사용해 CIF 문자열을 파일처럼 처리
            cif_io = StringIO(cif_string)
            atoms = read(cif_io, format='cif')
        return atoms
    except Exception as e:
        raise ValueError(f"CIF 파싱 오류: {e}")


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


def get_atoms_from_data(data: dict) -> Atoms:
    """
    데이터에서 atoms 또는 cif 정보를 추출하여 ASE Atoms 객체를 반환합니다.
    
    Args:
        data: 'atoms' 또는 'cif' key를 포함하는 딕셔너리
        
    Returns:
        ASE Atoms 객체
    """
    if 'atoms' in data and data['atoms'] is not None:
        # atoms 딕셔너리가 있는 경우
        return dict_to_atoms(convert_arrays_to_lists(data['atoms']))
    elif 'cif' in data and data['cif'] is not None:
        # cif 문자열이 있는 경우
        return cif_to_atoms(data['cif'])
    else:
        raise ValueError("데이터에 'atoms' 또는 'cif' 정보가 없습니다.")


def atom_to_torch_graph_data(
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


def atom_to_torch_graph_data_knn(
    atoms,
    k: int = 12,
    cutoff_start: float = 3.0,
    cutoff_step: float = 0.75,
    cutoff_max: float = 12.0,
    add_vectors: bool = True,
    eps: float = 1e-8,
):
    """k-NN 기반 그래프 생성을 atom_to_torch_graph_data 에 위임합니다."""
    atoms = dict_to_atoms(atoms) if isinstance(atoms, dict) else atoms
    return atom_to_torch_graph_data(
        atoms,
        cutoff=cutoff_max,
        add_vectors=add_vectors,
        k=k,
        cutoff_start=cutoff_start,
        cutoff_step=cutoff_step,
        cutoff_max=cutoff_max,
        eps=eps,
    )


def atom_to_orb_graph_data(atoms, system_config=None):
    # atoms = jatoms.ase_converter()
    return atomic_system.ase_atoms_to_atom_graphs(atoms, system_config=system_config, device='cpu')


class CLaCBaseDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            debug: bool = False,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.data_path = {
            'train': data_path+'_train.parquet',
            'val': data_path+'_val.parquet',
            'test': data_path+'_test.parquet',
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = CLaCTokenizer(tokenizer_model)
        self.debug = debug
        self._stage = 'train'
        # if tokenizer_model == 'facebook/galactica-125m':
        #     self.tokenizer.pad_token_id = 1
        #     self.tokenizer.mask_token_id = 3
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)
        
        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset

        # dataset = dataset['train'].train_test_split(test_size=0.995) if self.debug else dataset

        # dataset = dataset.shuffle()
        # dataset = dataset['train']#.map(self.preprocess_function, batched=False, num_proc=self.num_workers)
        # # split dataset
        # dataset = dataset.train_test_split(test_size=0.2)
        # train_dataset = dataset['train']
        # dataset = dataset['test']
        # dataset = dataset.train_test_split(test_size=0.5)
        # val_dataset = dataset['train']
        # test_dataset = dataset['test']
        # self.dataset = DatasetDict({
        #     'train': train_dataset,
        #     'val': val_dataset,
        #     'test': test_dataset
        # })
        # # self.dataset = dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    def train_dataloader(self):
        self._stage = 'train'
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        self._stage = 'val'
        return DataLoader(self.dataset['val'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        self._stage = 'test'
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    

class CLaCDataModule(CLaCBaseDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            graphdatatype: str = 'orb',  # 'torch_geometric' or 'orb'
            textdatatype: str = 'narratives',
            sentencewise: bool = False,
            use_visual_intramodal_loss: bool = True,
            use_textual_intramodal_loss: bool = True,
            replace_formula_prob: float = 0.0,  # Probability of replacing chemical formulas
            *args,
            **kwargs,
        ):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, *args, **kwargs)
        self.token_fn = lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=128)
        self.graphdatatype = graphdatatype
        if self.graphdatatype == 'orb':
            orbff = pretrained.orb_v3_direct_20_omat(
                device='cpu',  # or 'cuda'
                precision='float32-high'  # or 'float32-highest' / 'float64'
            )
            self.orbff_system_config = orbff.system_config
        self.textdatatype = textdatatype
        self.sentencewise = sentencewise

        # Store fine-grained intramodal loss settings
        self.use_visual_intramodal_loss = use_visual_intramodal_loss
        self.use_textual_intramodal_loss = use_textual_intramodal_loss

        # Compute whether we need augmentation (either visual or textual intramodal enabled)
        self.use_intramodal_loss = self.use_visual_intramodal_loss or self.use_textual_intramodal_loss

        # Chemical formula replacement settings
        self.replace_formula_prob = replace_formula_prob
        self.enable_formula_replacement = False  # Controlled manually during training

    def train_collate_fn(self, features: List[dict]):
        """Collate function for training with data augmentation."""
        # Independently augment graph and text based on their respective flags
        if self.use_visual_intramodal_loss:
            graph_batch1, graph_batch2 = self.graph_data_collator(features, augment=True)
        else:
            graph_batch1 = self.graph_data_collator(features, augment=False)
            graph_batch2 = None

        if self.use_textual_intramodal_loss:
            text_batch1, text_batch2 = self.text_data_collator(features, self.token_fn, augment=True)
        else:
            text_batch1 = self.text_data_collator(features, self.token_fn, augment=False)
            text_batch2 = None

        return graph_batch1, graph_batch2, text_batch1, text_batch2

    def eval_collate_fn(self, features: List[dict]):
        """Collate function for evaluation without data augmentation."""
        graph_batch = self.graph_data_collator(features, augment=False)
        text_batch = self.text_data_collator(features, self.token_fn, augment=False)
        return graph_batch, text_batch

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.train_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size, collate_fn=self.eval_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.eval_collate_fn, num_workers=self.num_workers)

    def graph_data_collator(self, features: List[Dict[str, Any]], augment: bool = True):
        """
        Collates graph data. If augment is True, creates two augmented versions of the graph.
        Otherwise, returns a single, non-augmented graph batch.
        """
        jatoms_list = [get_atoms_from_data(f) for f in features]
        
        if augment:
            if self.graphdatatype == 'torch_geometric':
                batch1 = Batch.from_data_list([atom_to_torch_graph_data(perturb_structure(atoms)) for atoms in jatoms_list])
                batch2 = Batch.from_data_list([atom_to_torch_graph_data(apply_strain(atoms)) for atoms in jatoms_list])
                return batch1, batch2
            elif self.graphdatatype == 'orb':
                batch1 = batch_graphs([atom_to_orb_graph_data(perturb_structure(atoms), self.orbff_system_config) for atoms in jatoms_list])
                batch2 = batch_graphs([atom_to_orb_graph_data(apply_strain(atoms), self.orbff_system_config) for atoms in jatoms_list])
                return batch1, batch2
        else:
            if self.graphdatatype == 'torch_geometric':
                batch = Batch.from_data_list([atom_to_torch_graph_data(atoms) for atoms in jatoms_list])
                return batch
            elif self.graphdatatype == 'orb':
                batch = batch_graphs([atom_to_orb_graph_data(atoms, self.orbff_system_config) for atoms in jatoms_list])
                return batch

    def text_data_collator(self, features: List[dict], token_fn, augment: bool = True):
        """
        Collates text data.
        If augment is True (training), it creates two augmented text batches by randomly dropping and shuffling sentences.
        If augment is False (evaluation), it returns a single batch containing the full text.
        """
        if augment:
            batch1 = []
            batch2 = []

            def create_augmented_text(sents: list, keep_prob: float = 0.5) -> str:
                # Randomly drop sentences
                kept_sents = [s for s in sents if random.random() < keep_prob]

                # If all sentences are dropped, keep at least one random one to avoid empty text.
                if not kept_sents and sents:
                    kept_sents = [random.choice(sents)]

                # Shuffle the order of kept sentences
                random.shuffle(kept_sents)

                return ' '.join(kept_sents)

            for f in features:
                if self.textdatatype == 'narratives':
                    text_source = ''.join(filter(None, [f.get('gpt_text'), f.get('gpt_explanation')]))
                    sentences = [sent.text for sent in nlp(text_source).sents]
                    if not sentences:
                        sentences = [text_source]
                elif self.textdatatype == 'papers':
                    paragraphs = f.get('text') or [[]]
                    paragraph = random.choice(paragraphs)
                    if not paragraph:
                        paragraph = [""]
                    sentences = [sent.text for para in paragraph for sent in nlp(para).sents]
                    if not sentences:
                        sentences = [' '.join(paragraph)]

                # Create two different augmentations from the same source sentences
                augmented_text_1 = create_augmented_text(sentences)
                augmented_text_2 = create_augmented_text(sentences)

                # Apply chemical formula replacement if enabled
                if self.enable_formula_replacement and random.random() < self.replace_formula_prob:
                    augmented_text_1 = replace_chemical_formula(augmented_text_1)
                if self.enable_formula_replacement and random.random() < self.replace_formula_prob:
                    augmented_text_2 = replace_chemical_formula(augmented_text_2)

                batch1.append(token_fn(augmented_text_1))
                batch2.append(token_fn(augmented_text_2))

            return default_data_collator(batch1), default_data_collator(batch2)
        
        else: # not augment (evaluation)
            batch = []
            for f in features:
                if self.textdatatype == 'narratives':
                    parts = [f.get('gpt_text'), f.get('gpt_explanation')]
                    full_text = ' '.join(p for p in parts if p)
                elif self.textdatatype == 'papers':
                    paragraphs = f.get('text') or []
                    flattened = [para for paragraph in paragraphs for para in paragraph]
                    full_text = ' '.join(flattened)
                else:
                    full_text = '' # Should not happen with current configs
                
                encoded = token_fn(full_text)
                batch.append(encoded)
                
            return default_data_collator(batch)


class GraphSupervisedDataModule(CLaCBaseDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            label: str = 'y',
            task: str = 'classification',
            graphdatatype: str = 'torch_geometric',  # 'torch_geometric' or 'orb'
            *args,
            **kwargs,
        ):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, *args, **kwargs)
        # self.graph_data_collator = graph_data_collator
        self.label = label
        self.task = task
        self.graphdatatype = graphdatatype

        # Initialize ORB system config if needed
        if self.graphdatatype == 'orb':
            orbff = pretrained.orb_v3_direct_20_omat(
                device='cpu',
                precision='float32-high'
            )
            self.orbff_system_config = orbff.system_config

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)

        # Filter out samples with NA/null labels for regression/classification
        print(f"\n[DataModule] Original dataset sizes:")
        for split in ['train', 'val', 'test']:
            if split in dataset:
                print(f"  {split}: {len(dataset[split])} samples")

        # Filter out rows where label is None/NaN
        def filter_valid_labels(example):
            label_val = example.get(self.label)
            if label_val is None:
                return False
            # Check for NaN (float)
            if isinstance(label_val, float):
                return not (label_val != label_val)  # NaN != NaN is True
            # Check for string representations of missing values
            if isinstance(label_val, str):
                return label_val.lower() not in ['nan', 'none', 'na', 'n/a', '']
            return True

        for split in ['train', 'val', 'test']:
            if split in dataset:
                dataset[split] = dataset[split].filter(filter_valid_labels)

        print(f"\n[DataModule] After filtering NA labels:")
        for split in ['train', 'val', 'test']:
            if split in dataset:
                print(f"  {split}: {len(dataset[split])} samples")

        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self.categories = dataset['train'].unique(self.label)
            self.label_encoder = LabelEncoder().fit(self.categories)

        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        if self.task == 'classification':
            for f in features:
                f['y'] = self.label_encoder.transform([f[self.label]])[0]
        elif self.task == 'regression':
            for f in features:
                f['y'] = float(f[self.label])
        graph_batch = self.graph_data_collator(features)
        return graph_batch

    def graph_data_collator(self, features: List[dict]) -> Dict[str, Any]:
        """
        Collate graph data from atoms or precomputed node_feat/edge_index.
        """
        # Check if data has atoms (new format) or node_feat (old format)
        if 'atoms' in features[0] or 'cif' in features[0]:
            # New format: atoms dict -> convert to graph
            jatoms_list = [get_atoms_from_data(f) for f in features]

            if self.graphdatatype == 'torch_geometric':
                data_list = []
                for i, atoms in enumerate(jatoms_list):
                    graph_data = atom_to_torch_graph_data(atoms)
                    # Use long for classification, float for regression
                    dtype = torch.long if self.task == 'classification' else torch.float32
                    graph_data.y = torch.tensor([features[i]['y']], dtype=dtype)
                    data_list.append(graph_data)
                return Batch.from_data_list(data_list)
            elif self.graphdatatype == 'orb':
                graph_list = [atom_to_orb_graph_data(atoms, self.orbff_system_config) for atoms in jatoms_list]
                batch = batch_graphs(graph_list)
                # Add y labels to batch
                # Use long for classification, float for regression
                dtype = torch.long if self.task == 'classification' else torch.float32
                batch.y = torch.tensor([f['y'] for f in features], dtype=dtype)
                return batch
        else:
            # Old format: node_feat/edge_index already computed
            # Use long for classification, float for regression
            y_dtype = torch.long if self.task == 'classification' else torch.float32
            return Batch.from_data_list([Data(x=torch.tensor(f["node_feat"], dtype=torch.float32),
                                            edge_index=torch.tensor(f['edge_index']),
                                            edge_attr=torch.tensor(f['edge_attr'], dtype=torch.float32),
                                            y=torch.tensor(f['y'], dtype=y_dtype)) for f in features])

class QuestionEvaluationDataModule(CLaCBaseDataModule):
    def __init__(self,
                 data_path: str,
                 batch_size: int = 16,
                 num_workers: int = 12,
                 tokenizer_model: str = 'bert-base-uncased',
                 debug: bool = False,
                 label: str = 'structure_question_list',
                 graphdatatype: str = 'torch_geometric',
                 *args,
                 **kwargs):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, debug, *args, **kwargs)
        # self.graph_data_collator = graph_data_collator
        # self.text_data_collator = question_batch_data_collator
        self.label = label
        self.graphdatatype = graphdatatype
        self.token_fn = lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=512)

        # Initialize ORB system config if needed
        if self.graphdatatype == 'orb':
            orbff = pretrained.orb_v2(
                device='cpu',
                precision='float32-high'
            )
            self.orbff_system_config = orbff.system_config
        else:
            self.orbff_system_config = None

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = self.graph_data_collator(features)
        text_batch = self.question_batch_data_collator(features, self.token_fn, self.label)
        return graph_batch, text_batch
    
    def graph_data_collator(self, features: List[dict]) -> Dict[str, Any]:
        """
        Collate graph data. Supports both torch_geometric and orb graphdatatypes.
        Note: y labels are not used in QA evaluation, so they are not assigned.
        """
        # Check if data has atoms/cif (new format) or node_feat (old format)
        if 'atoms' in features[0] or 'cif' in features[0]:
            # New format: convert atoms to graphs
            jatoms_list = [get_atoms_from_data(f) for f in features]

            if self.graphdatatype == 'torch_geometric':
                data_list = []
                for i, atoms in enumerate(jatoms_list):
                    graph_data = atom_to_torch_graph_data(atoms)
                    # Note: y labels not needed for QA evaluation
                    data_list.append(graph_data)
                return Batch.from_data_list(data_list)
            elif self.graphdatatype == 'orb':
                graph_list = [atom_to_orb_graph_data(atoms, self.orbff_system_config) for atoms in jatoms_list]
                batch = batch_graphs(graph_list)
                # Note: y labels not needed for QA evaluation (and AtomGraphs doesn't support direct attribute assignment)
                return batch
        else:
            # Old format: node_feat/edge_index already computed (torch_geometric only)
            # Note: y labels kept for backward compatibility with old dataset format
            return Batch.from_data_list([Data(x=torch.tensor(f["node_feat"], dtype=torch.float32),
                                            edge_index=torch.tensor(f['edge_index']),
                                            edge_attr=torch.tensor(f['edge_attr'], dtype=torch.float32),
                                            y=torch.tensor(f['y'], dtype=torch.float32)) for f in features])

    def question_batch_data_collator(self, features: List[dict], token_fn, label) -> Dict[str, Any]:
        '''
        '''
        encoded_per_example = [default_data_collator([token_fn(q) for q in f[label]]) for f in features]

        if not encoded_per_example:
            raise ValueError("Received an empty batch in question_batch_data_collator.")

        stacked_batch = {}
        for key in encoded_per_example[0].keys():
            stacked_batch[key] = torch.stack([encoded[key] for encoded in encoded_per_example], dim=0)

        # For zero-shot QA: the correct answer is always the first choice (index 0)
        # No need to use _argmax_index since the answer is always 0
        labels = torch.zeros(len(features), dtype=torch.long)
        choice_sizes = torch.tensor([encoded['input_ids'].size(0) for encoded in encoded_per_example], dtype=torch.long)

        stacked_batch['labels'] = labels
        stacked_batch['choice_sizes'] = choice_sizes
        return stacked_batch
