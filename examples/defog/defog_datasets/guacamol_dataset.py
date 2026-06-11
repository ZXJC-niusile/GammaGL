import os
import os.path as osp
import numpy as np
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
)


class GuacaMolDataset(InMemoryDataset):
    r"""The GuacaMol dataset for molecular graph generation.

    From the `"GuacaMol: Benchmarking Models for de Novo Molecular Design"
    <https://arxiv.org/abs/1811.09621>`_ benchmark.
    Contains ~1.6M molecules from ChEMBL.

    Requires **RDKit** (``pip install rdkit``).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    URLS = {
        'train': 'https://figshare.com/ndownloader/files/13612760',
        'test': 'https://figshare.com/ndownloader/files/13612757',
        'valid': 'https://figshare.com/ndownloader/files/13612766',
    }

    ATOM_DECODER = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S',
                    'Se', 'Si']
    ATOM_ENCODER = {atom: i for i, atom in enumerate(ATOM_DECODER)}
    NUM_ATOM_TYPES = 12
    NUM_EDGE_TYPES = 5  # no-bond=0, single=1, double=2, triple=3, aromatic=4

    STATS = {
        'valencies': [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4],
        'atom_weights': {0: 12, 1: 14, 2: 16, 3: 19, 4: 10.8, 5: 79.9,
                         6: 35.4, 7: 126.9, 8: 31, 9: 32, 10: 79, 11: 28.1},
        'max_weight': 800.0,
        'num_node_types': 12,
        'num_edge_types': 5,
        'node_types': np.array([
            7.409e-01, 1.069e-01, 1.122e-01, 1.421e-02, 6.058e-05,
            1.717e-03, 8.411e-03, 2.290e-04, 5.695e-04, 1.467e-02,
            4.153e-05, 5.342e-05,
        ]),
        'edge_types': np.array([
            9.253e-01, 3.624e-02, 4.849e-03, 1.651e-04, 3.349e-02,
        ]),
        'n_nodes': np.array([
            0, 0, 3.5760e-06, 2.7893e-05, 6.9374e-05, 1.6020e-04,
            2.8036e-04, 4.3484e-04, 7.3022e-04, 1.1722e-03,
            1.7830e-03, 2.8129e-03, 4.0981e-03, 5.5421e-03,
            7.9645e-03, 1.0824e-02, 1.4459e-02, 1.8818e-02,
            2.3961e-02, 2.9558e-02, 3.6324e-02, 4.1931e-02,
            4.8105e-02, 5.2316e-02, 5.6601e-02, 5.7483e-02,
            5.6685e-02, 5.2317e-02, 5.2107e-02, 4.9651e-02,
            4.8100e-02, 4.4363e-02, 4.0704e-02, 3.5719e-02,
            3.1685e-02, 2.6821e-02, 2.2542e-02, 1.8591e-02,
            1.6114e-02, 1.3399e-02, 1.1543e-02, 9.6116e-03,
            8.4744e-03, 6.9532e-03, 6.2001e-03, 4.9921e-03,
            4.4378e-03, 3.5803e-03, 3.3078e-03, 2.7085e-03,
            2.6784e-03, 2.2050e-03, 2.0533e-03, 1.5598e-03,
            1.5177e-03, 9.8626e-04, 8.6396e-04, 5.6429e-04,
            5.0422e-04, 2.9323e-04, 2.2243e-04, 9.8697e-05,
            9.9413e-05, 6.0077e-05, 6.9374e-05, 3.0754e-05,
            3.5045e-05, 1.6450e-05, 2.1456e-05, 1.2874e-05,
            1.2158e-05, 5.7216e-06, 7.1520e-06, 2.8608e-06,
            2.8608e-06, 7.1520e-07, 2.8608e-06, 1.4304e-06,
            7.1520e-07, 0, 0, 0, 7.1520e-07, 0, 1.4304e-06,
            7.1520e-07, 7.1520e-07, 0, 1.4304e-06,
        ], dtype=np.float32),
        'max_n_nodes': 88,
        'valency_distribution': np.pad(
            np.array(
                [0.0, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053],
                dtype=np.float32,
            ),
            (0, 3 * 88 - 2 - 7),
        ),
    }

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = 'guacamol_gen'
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.smiles', 'test.smiles', 'valid.smiles']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        # Try Figshare direct download first
        for split, filename in [('train', 'train.smiles'),
                                ('test', 'test.smiles'),
                                ('valid', 'valid.smiles')]:
            download_url(self.URLS[split], self.raw_dir, filename=filename)

        # Check if files are valid (non-empty)
        all_valid = True
        for filename in self.raw_file_names:
            path = osp.join(self.raw_dir, filename)
            if not osp.exists(path) or osp.getsize(path) == 0:
                all_valid = False
                break

        if not all_valid:
            # Figshare may be blocked by WAF; fallback to guacamol package data generation
            print("Figshare download returned empty files (likely WAF blocked).")
            print("Attempting to generate GuacaMol data via the official guacamol package...")
            try:
                self._generate_via_guacamol_package()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download GuacaMol data from Figshare and could not "
                    f"generate via guacamol package: {e}. "
                    f"Please install guacamol (pip install guacamol) or manually "
                    f"download the .smiles files from Figshare and place them in "
                    f"{self.raw_dir}"
                ) from e

    def _generate_via_guacamol_package(self):
        """Generate train/valid/test .smiles via the official guacamol package.

        This is a fallback when Figshare direct downloads are blocked by WAF.
        It downloads ChEMBL 24.1 from EBI FTP and runs the canonical filtering
        pipeline.  The resulting files are copied into ``self.raw_dir``.
        """
        import tempfile
        import shutil
        import importlib.util

        spec = importlib.util.find_spec('guacamol')
        if spec is None:
            raise RuntimeError("guacamol package not installed")

        get_data_path = osp.join(osp.dirname(spec.origin), 'data', 'get_data.py')
        if not osp.exists(get_data_path):
            raise RuntimeError(f"guacamol get_data.py not found at {get_data_path}")

        tmpdir = tempfile.mkdtemp(prefix='guacamol_datagen_')
        try:
            import subprocess
            import sys
            cmd = [
                sys.executable, get_data_path,
                '--destination', tmpdir,
                '--n_jobs', '8',
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"guacamol data generation failed:\n{result.stderr}"
                )
            # Copy generated files with expected names
            mapping = {
                'chembl24_canon_train.smiles': 'train.smiles',
                'chembl24_canon_dev-valid.smiles': 'valid.smiles',
                'chembl24_canon_test.smiles': 'test.smiles',
            }
            for src_name, dst_name in mapping.items():
                src = osp.join(tmpdir, src_name)
                dst = osp.join(self.raw_dir, dst_name)
                if osp.exists(src):
                    shutil.copy2(src, dst)
                    print(f"Copied {src_name} -> {dst_name}")
                else:
                    raise RuntimeError(f"Expected output {src} not found")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def process(self):
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
        except ImportError:
            raise ImportError(
                "GuacaMolDataset requires RDKit. Install via: pip install rdkit"
            )

        RDLogger.DisableLog('rdApp.*')

        atom_encoder = self.ATOM_ENCODER
        bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3,
                      BT.AROMATIC: 4}
        num_bond_types = self.NUM_EDGE_TYPES
        num_atom_types = self.NUM_ATOM_TYPES

        split_files = {
            'train': 'train.smiles',
            'val': 'valid.smiles',
            'test': 'test.smiles',
        }

        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            smiles_path = osp.join(self.raw_dir, split_files[split_name])
            with open(smiles_path, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]

            data_list = []
            for smile in smiles_list:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    continue

                n_atoms = mol.GetNumAtoms()
                if n_atoms == 0:
                    continue

                # Node features: one-hot atom type
                type_idx = []
                valid = True
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol not in atom_encoder:
                        valid = False
                        break
                    type_idx.append(atom_encoder[symbol])
                if not valid:
                    continue

                x = np.zeros((n_atoms, num_atom_types), dtype=np.float32)
                for j, t in enumerate(type_idx):
                    x[j, t] = 1.0

                # Edge index and attributes
                rows, cols, edge_feats = [], [], []
                for bond in mol.GetBonds():
                    start = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    bt = bond.GetBondType()
                    if bt not in bond_types:
                        continue
                    bond_idx = bond_types[bt]
                    for s, d in [(start, end), (end, start)]:
                        rows.append(s)
                        cols.append(d)
                        feat = np.zeros(num_bond_types, dtype=np.float32)
                        feat[bond_idx] = 1.0
                        edge_feats.append(feat)

                if len(rows) > 0:
                    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
                    edge_attr = np.stack(edge_feats, axis=0)
                else:
                    continue

                y = np.zeros((1, 0), dtype=np.float32)

                data = Graph(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    n_nodes=np.array([n_atoms], dtype=np.int64),
                    to_tensor=True,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            if len(data_list) == 0:
                raise RuntimeError(
                    f"GuacaMolDataset: no valid graphs generated for split "
                    f"'{split_name}'. The raw SMILES file may be empty or "
                    f"corrupt. Please check {self.raw_dir}"
                )

            collated_data, slices = self.collate(data_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def get_stats(self):
        """Return pre-computed dataset statistics for DeFoG training."""
        return self.STATS.copy()

    def __repr__(self) -> str:
        return f'GuacaMolDataset({len(self)})'
