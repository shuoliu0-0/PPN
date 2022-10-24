"""
The dataset used in training GROVER.
"""
import math
import os
import csv
from typing import Union, List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

import grover.util.utils as feautils
from grover.data import mol2graph
from grover.data.moldataset import MoleculeDatapoint
from grover.data.task_labels import atom_to_vocab, bond_to_vocab


def get_data(data_path, logger=None):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
    :return:
    """
    debug = logger.debug if logger is not None else print
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    feature_path = os.path.join(data_path, "feature")

    fin = open(summary_path)
    n_files = int(fin.readline().strip().split(":")[-1])
    n_samples = int(fin.readline().strip().split(":")[-1])
    sample_per_file = int(fin.readline().strip().split(":")[-1])
    debug("Loading data:")
    debug("Number of files: %d" % n_files)
    debug("Number of samples: %d" % n_samples)
    debug("Samples/file: %d" % sample_per_file)

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        feature_path_i = os.path.join(feature_path, str(i) + ".npz")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint(smiles_path_i, feature_path_i, n_samples_i))
    return BatchMolDataset(datapoints), sample_per_file


def split_data(data,
               split_type='random',
               sizes=(0.8, 0.1, 0.1),
               seed=0,
               logger=None):
    """
    Split data with given train/validation/test ratio.
    :param data:
    :param split_type:
    :param sizes:
    :param seed:
    :param logger:
    :return:
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == "random":
        data.shuffle(seed=seed)
        data = data.data

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return BatchMolDataset(train), BatchMolDataset(val), BatchMolDataset(test)
    else:
        raise NotImplementedError("Do not support %s splits" % split_type)


class BatchDatapoint:
    def __init__(self,
                 smiles_file,
                 feature_file,
                 n_samples,
                 ):
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        features = self.load_feature()
        self.datapoints = []

        with open(self.smiles_file) as f:
            reader = csv.reader(f)
            next(reader)
            for i, line in enumerate(reader):
                # line = line[0]
                d = MoleculeDatapoint(line=line,
                                      features=features[i])
                self.datapoints.append(d)

        assert len(self.datapoints) == self.n_samples

    def load_feature(self):
        return feautils.load_features(self.feature_file)

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchMolDataset(Dataset):
    def __init__(self, data: List[BatchDatapoint],
                 graph_per_file=None):
        self.data = data

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        # print(idx)
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        return self.data[dp_idx][real_idx]

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res


class GroverCollator(object):
    def __init__(self, shared_dict, atom_vocab, bond_vocab, graph_vocab, feature_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.graph_vocab = graph_vocab
        self.feature_vocab = feature_vocab

    def get_atom_feature(self, mol):
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')  # get feature library
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)  # Building Feature Factories
        Hydrophobe_feats = (
            factory.GetFeaturesForMol(mol, includeOnly="Hydrophobe"))  # Search for features using feature factories
        acceptor_feats = (factory.GetFeaturesForMol(mol, includeOnly="Acceptor"))
        donor_feats = (factory.GetFeaturesForMol(mol, includeOnly="Donor"))
        NegIonizable_feats = (factory.GetFeaturesForMol(mol, includeOnly="NegIonizable"))
        PosIonizable_feats = (factory.GetFeaturesForMol(mol, includeOnly="PosIonizable"))
        feats_id = {'Hydrophobe': [list(sublist.GetAtomIds()) for sublist in Hydrophobe_feats],
                    'acceptor': [list(sublist.GetAtomIds()) for sublist in acceptor_feats],
                    'donor': [list(sublist.GetAtomIds()) for sublist in donor_feats],
                    'NegIonizable': [list(sublist.GetAtomIds()) for sublist in NegIonizable_feats],
                    'PosIonizable': [list(sublist.GetAtomIds()) for sublist in PosIonizable_feats]}
        return feats_id

    def atom_feature_mask0(self, smiles_batch):
        vocab_label = [0]
        a = self.feature_vocab.stoi.get('acceptor0-donor0-NegIonizable0-PosIonizable0-Hydrophobe0', self.feature_vocab.other_index)
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            feature_id = self.get_atom_feature(mol)
            label_dict = {}
            for atom in mol.GetAtoms():
                p = atom.GetIdx()
                v = 'acceptor' + ['1' if [atom.GetIdx()] in feature_id['acceptor'] else '0'][0] + \
                    '-donor' + ['1' if [atom.GetIdx()] in feature_id['donor'] else '0'][0] + \
                    '-NegIonizable' + ['1' if [atom.GetIdx()] in feature_id['NegIonizable'] else '0'][0] + \
                    '-PosIonizable' + ['1' if [atom.GetIdx()] in feature_id['PosIonizable'] else '0'][0] + \
                    '-Hydrophobe' + ['1' if [atom.GetIdx()] in feature_id['Hydrophobe'] else '0'][0]
                mlabel[p] = self.feature_vocab.stoi.get(v, self.feature_vocab.other_index)
                if mlabel[p] not in label_dict:
                    label_dict[mlabel[p]] = [p]
                else:
                    label_dict[mlabel[p]].append(p)

            m = label_dict.get(a)
            if m is not None and len(label_dict) > 1:
                m1 = max(len(v) for v in label_dict.values())
                if len(m) == m1:
                    m2 = min(len(v) for v in label_dict.values() if v != m)
                    if m1>m2:
                        perm = np.random.permutation(m)[:(m1-m2)]
                        for i in perm:
                            mlabel[i] = 0
            vocab_label.extend(mlabel)
        return vocab_label

        
    def atom_feature_mask(self, smiles_batch):
        vocab_label = [0]
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            feature_id = self.get_atom_feature(mol)
            label_dict = {}
            for atom in mol.GetAtoms():
                p = atom.GetIdx()
                v = 'acceptor' + ['1' if [atom.GetIdx()] in feature_id['acceptor'] else '0'][0] + \
                    '-donor' + ['1' if [atom.GetIdx()] in feature_id['donor'] else '0'][0] + \
                    '-NegIonizable' + ['1' if [atom.GetIdx()] in feature_id['NegIonizable'] else '0'][0] + \
                    '-PosIonizable' + ['1' if [atom.GetIdx()] in feature_id['PosIonizable'] else '0'][0] + \
                    '-Hydrophobe' + ['1' if [atom.GetIdx()] in feature_id['Hydrophobe'] else '0'][0] + \
                    '-Aromatic' + ['1' if atom.GetIsAromatic() else '0'][0]
                mlabel[p] = self.feature_vocab.stoi.get(v, self.feature_vocab.other_index)
                lenx = 0
                # for idx in [8, 15, 29, 43, 55]:
                for idx in [8, 15, 29, 43, 55, 65]:
                    if v[idx] == '1':
                        if idx==65 and lenx==0:
                            continue
                        lenx += 1
                if lenx not in label_dict:
                    label_dict[lenx] = [p]
                else:
                    label_dict[lenx].append(p)

            # m0 = [len(label_dict.get(0)) if label_dict.get(0) is not None else 0][0]
            # m1 = [len(label_dict.get(1)) if label_dict.get(1) is not None else 0][0]
            # m2 = [len(label_dict.get(2)) if label_dict.get(2) is not None else 0][0]
            # m3 = [len(label_dict.get(3)) if label_dict.get(3) is not None else 0][0]
            # s = math.exp(m0) + math.exp(m1) + math.exp(m2) + math.exp(m3)

            s = 0
            for key in label_dict.keys():
                s += math.exp(key)
            for key,value in label_dict.items():
                if key>=3:
                    sample = len(value)
                else:
                    # sample = math.exp(2*m2*key)/s
                    sample = math.exp(key)/s
                    # if sample<m2:
                    #     sample = m2
                task = int(len(value)*(1-sample))
                perm = np.random.permutation(value)[:task]
                for i in perm:
                    mlabel[i] = 0
            vocab_label.extend(mlabel)
        return vocab_label

        
    def atom_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on atoms.
        :param smiles_batch:
        :return: The corresponding atom labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def bond_random_mask(self, smiles_batch):
        """
        Perform the random mask operaiion on bonds.
        :param smiles_batch:
        :return: The corresponding bond labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            nm_atoms = mol.GetNumAtoms()
            nm_bonds = mol.GetNumBonds()
            mlabel = []
            n_mask = math.ceil(nm_bonds * percent)
            perm = np.random.permutation(nm_bonds)[:n_mask]
            virtual_bond_id = 0
            for a1 in range(nm_atoms):
                for a2 in range(a1 + 1, nm_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if virtual_bond_id in perm:
                        label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            # todo: might need to consider bond_drop_rate
            # todo: double check reverse bond
            vocab_label.extend(mlabel)
        return vocab_label

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        atom_vocab_label = torch.Tensor(self.atom_random_mask(smiles_batch)).long()
        bond_vocab_label = torch.Tensor(self.bond_random_mask(smiles_batch)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        atom_feature_label = torch.Tensor(self.atom_feature_mask(smiles_batch)).long()
        # may be some mask here
        res = {"graph_input": batchgraph,
               "targets": {"av_task": atom_vocab_label,
                           "bv_task": bond_vocab_label,
                           "fg_task": fgroup_label,
                           "af_task": atom_feature_label}
               }
        return res
