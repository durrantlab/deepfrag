"""
Contains utility code for reading packed data files.
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import tqdm

# Atom typing
# 
# Atom typing is the process of figuring out which layer each atom should be
# written to. For ease of testing, the packed data file contains a lot of
# potentially useful atomic information which can be distilled during the
# data loading process.
# 
# Atom typing is implemented by map functions of the type:
#   (atom descriptor) -> (layer index)
# 
# If the layer index is -1, the atom is ignored.

def rec_typer_single(t):
    num, hacc, hdon, aro, _ = t
    
    if num != 1:
        return 0
    else:
        return -1
    
def rec_typer_single_h(t):
    return 0
    
def rec_typer_basic(t):
    BASIC_TYPES = [6,7,8,16]
    num, hacc, hdon, aro, _ = t
    
    if num in BASIC_TYPES:
        return BASIC_TYPES.index(num)
    else:
        return -1
    
def rec_typer_basic_h(t):
    BASIC_TYPES = [1,6,7,8,16]
    num, hacc, hdon, aro, _ = t
    
    if num in BASIC_TYPES:
        return BASIC_TYPES.index(num)
    else:
        return -1

# aro, hdon, hacc
REC_DESC = [
    (6,0,0,0), # C
    (6,1,0,0), # C-aro
    (7,0,0,0), # N
    (7,0,0,1), # N-hacc
    (7,0,1,0), # N-hdon
    (7,1,0,0), # N-aro
    (7,0,1,1), # N-hdon-hacc
    (7,1,0,1), # N-aro-hacc
    (7,1,1,0), # N-aro-hdon
    (8,0,0,0), # O
    (8,0,0,1), # O-hacc
    (8,0,1,1), # O-hdon-hacc
    (16,0,0,0), #S
]

REC_DESC_H = [
    (1,0,0,0), # H
    (6,0,0,0), # C
    (6,1,0,0), # C-hacc
    (7,0,0,0), # N
    (7,0,0,1), # N-aro
    (7,0,1,0), # N-hdon
    (7,1,0,0), # N-hacc
    (7,0,1,1), # N-hdon-aro
    (7,1,0,1), # N-hacc-aro
    (7,1,1,0), # N-hacc-hdon
    (8,0,0,0), # O
    (8,0,0,1), # O-aro
    (8,0,1,1), # O-hdon-aro
    (16,0,0,0), #S
]
    
def rec_typer_desc(t):
    num, hacc, hdon, aro, _ = t
    
    f = (num,hacc,hdon,aro)
    
    if f in REC_DESC:
        return REC_DESC.index(f)
    else:
        return -1
    
def rec_typer_desc_h(t):
    num, hacc, hdon, aro, _ = t
    
    f = (num,hacc,hdon,aro)
    
    if f in REC_DESC_H:
        return REC_DESC_H.index(f)
    else:
        return -1
    
def lig_typer_single(t):
    if t != 1:
        return 0
    else:
        return -1

def lig_typer_single_h(t):
    return 0

def lig_typer_simple(t):
    BASIC_TYPES = [6,7,8]
    
    if t in BASIC_TYPES:
        return BASIC_TYPES.index(t)
    else:
        return -1
    
def lig_typer_simple_h(t):
    BASIC_TYPES = [1,6,7,8]
    
    if t in BASIC_TYPES:
        return BASIC_TYPES.index(t)
    else:
        return -1
    
def lig_typer_desc(t):
    DESC_TYPES = [6,7,8,16,9,15,17,35,5,53]
    
    if t in DESC_TYPES:
        return DESC_TYPES.index(t)
    else:
        return -1
    
def lig_typer_desc_h(t):
    DESC_TYPES = [1,6,7,8,16,9,15,17,35,5,53]
    
    if t in DESC_TYPES:
        return DESC_TYPES.index(t)
    else:
        return -1

REC_TYPER = {
    'single': rec_typer_single,
    'single_h': rec_typer_single_h,
    'simple': rec_typer_basic,
    'simple_h': rec_typer_basic_h,
    'desc': rec_typer_desc,
    'desc_h': rec_typer_desc_h
}

LIG_TYPER = {
    'single': lig_typer_single,
    'single_h': lig_typer_single_h,
    'simple': lig_typer_simple,
    'simple_h': lig_typer_simple_h,
    'desc': lig_typer_desc,
    'desc_h': lig_typer_desc_h
}


class FragmentDataset(Dataset):
    """Utility class to work with the packed fragments.h5 format."""
    
    def __init__(self, fragment_file, rec_typer, lig_typer, filter_rec=None,
                 fdist_min=None, fdist_max=None, fmass_min=None,
                 fmass_max=None, verbose=False, skip_remap=False):
        """Initializes the fragment dataset.
        
        Args:
            fragment_file: path to fragments.h5
            rec_typer: function to map receptor rows to layer index
            lig_typer: function to map ligand rows to layer index
            filter_rec: list of receptor ids to use (or None to use all)
            skip_remap: if True, don't prepare atom type information

        (filtering options):
            fdist_min: minimum fragment distance
            fdist_max: maximum fragment distance
            fmass_min: minimum fragment mass (Da)
            fmass_max: maximum fragment mass (Da)
        """
        self.verbose = verbose
        self._skip_remap = skip_remap

        self.rec = self._load_rec(fragment_file, rec_typer)
        self.frag = self._load_fragments(fragment_file, lig_typer)

        self.valid_idx = self._get_valid_examples(
            filter_rec, fdist_min, fdist_max, fmass_min, fmass_max)

    def _load_rec(self, fragment_file, rec_typer):
        """Loads receptor information."""
        f = h5py.File(fragment_file, 'r')
        
        rec_coords = f['rec_coords'][()]
        rec_types = f['rec_types'][()]
        rec_lookup = f['rec_lookup'][()]
        
        r = range(len(rec_types))
        if self.verbose:
            r = tqdm.tqdm(r, desc='Remap receptor atoms')

        rec_remapped = np.zeros(len(rec_types)).astype(np.int32)
        if not self._skip_remap:
            for i in r:
                rec_remapped[i] = rec_typer(rec_types[i])

        # create rec mapping
        rec_mapping = {}
        for i in range(len(rec_lookup)):
            rec_mapping[rec_lookup[i][0].decode('ascii')] = i
        
        rec = {
            'rec_coords': rec_coords,
            'rec_types': rec_types,
            'rec_remapped': rec_remapped,
            'rec_lookup': rec_lookup,
            'rec_mapping': rec_mapping
        }
        
        f.close()
        
        return rec
        
    def _load_fragments(self, fragment_file, lig_typer):
        """Loads fragment information."""
        f = h5py.File(fragment_file, 'r')

        frag_data = f['frag_data'][()]
        frag_lookup = f['frag_lookup'][()]
        frag_smiles = f['frag_smiles'][()]
        frag_mass = f['frag_mass'][()]
        frag_dist = f['frag_dist'][()]
        
        # unpack frag data into separate structures
        frag_coords = frag_data[:,:3].astype(np.float32)
        frag_types = frag_data[:,3].astype(np.int32)
        
        frag_remapped = None
        if self._skip_remap:
            frag_remapped = np.zeros(len(frag_types))
        else:
            frag_remapped = np.vectorize(lig_typer)(frag_types)
        
        # find and save connection point
        r = range(len(frag_lookup))
        if self.verbose:
            r = tqdm.tqdm(r, desc='Frag connection point')

        frag_conn = np.zeros((len(frag_lookup), 3))
        for i in r:
            _,f_start,f_end,_,_ = frag_lookup[i]
            fdat = frag_data[f_start:f_end]
            
            found = False
            for j in range(len(fdat)):
                if fdat[j][3] == 0:
                    frag_conn[i,:] = tuple(fdat[j])[:3]
                    found = True
                    break

            assert found, "missing fragment connection point at %d" % i
        
        frag = {
            'frag_coords': frag_coords,     # d_idx -> (x,y,z)
            'frag_types': frag_types,       # d_idx -> (type)
            'frag_remapped': frag_remapped, # d_idx -> (layer)
            'frag_lookup': frag_lookup,     # f_idx -> (rec_id, fstart, fend, pstart, pend)
            'frag_conn': frag_conn,         # f_idx -> (x,y,z)
            'frag_smiles': frag_smiles,     # f_idx -> smiles
            'frag_mass': frag_mass,         # f_idx -> mass
            'frag_dist': frag_dist,         # f_idx -> dist
        }
        
        f.close()

        return frag

    def _get_valid_examples(self, filter_rec, fdist_min, fdist_max, fmass_min,
                            fmass_max):
        """Returns an array of valid fragment indexes.
        
        "Valid" in this context means the fragment belongs to a receptor in
        filter_rec and the fragment abides by the optional mass/distance
        constraints.
        """
        # keep track of valid examples
        valid_mask = np.ones(self.frag['frag_lookup'].shape[0]).astype(np.bool)
        
        # filter by receptor id
        if filter_rec is not None:
            valid_rec = np.vectorize(lambda k: k[0].decode('ascii') in filter_rec)(self.frag['frag_lookup'])
            valid_mask *= valid_rec
            
        # filter by fragment distance
        if fdist_min is not None:
            valid_mask[self.frag['frag_dist'] < fdist_min] = 0
            
        if fdist_max is not None:
            valid_mask[self.frag['frag_dist'] > fdist_max] = 0
            
        # filter by fragment mass
        if fmass_min is not None:
            valid_mask[self.frag['frag_mass'] < fmass_min] = 0

        if fmass_max is not None:
            valid_mask[self.frag['frag_mass'] > fmass_max] = 0

        # convert to a list of indexes
        valid_idx = np.where(valid_mask)[0]

        return valid_idx

    def __len__(self):
        """Returns the number of valid fragment examples."""
        return self.valid_idx.shape[0]
    
    def __getitem__(self, idx):
        """Returns the Nth example.
        
        Returns a dict with:
            f_coords: fragment coordinates (Fx3)
            f_types: fragment layers (Fx1)
            p_coords: parent coordinates (Px3)
            p_types: parent layers (Px1)
            r_coords: receptor coordinates (Rx3)
            r_types: receptor layers (Rx1)
            conn: fragment connection point in the parent molecule (x,y,z)
            smiles: fragment smiles string
        """
        # convert to fragment index
        frag_idx = self.valid_idx[idx]
        
        # lookup fragment
        rec_id, f_start, f_end, p_start, p_end = self.frag['frag_lookup'][frag_idx]
        smiles = self.frag['frag_smiles'][frag_idx].decode('ascii')
        conn = self.frag['frag_conn'][frag_idx]
        
        # lookup receptor
        rec_idx = self.rec['rec_mapping'][rec_id.decode('ascii')]
        _, r_start, r_end = self.rec['rec_lookup'][rec_idx]
        
        # fetch data
        f_coords = self.frag['frag_coords'][f_start:f_end]
        f_types = self.frag['frag_remapped'][f_start:f_end]
        p_coords = self.frag['frag_coords'][p_start:p_end]
        p_types = self.frag['frag_remapped'][p_start:p_end]
        r_coords = self.rec['rec_coords'][r_start:r_end]
        r_types = self.rec['rec_remapped'][r_start:r_end]

        return {
            'f_coords': f_coords,
            'f_types': f_types,
            'p_coords': p_coords,
            'p_types': p_types,
            'r_coords': r_coords,
            'r_types': r_types,
            'conn': conn,
            'smiles': smiles
        }

    def get_valid_smiles(self):
        """Returns a list of all valid smiles fragments."""
        valid_smiles = set()

        for idx in self.valid_idx:
            smiles = self.frag['frag_smiles'][idx].decode('ascii')
            valid_smiles.add(smiles)

        return list(valid_smiles)


class FingerprintDataset(Dataset):

    def __init__(self, fingerprint_file):
        """Initializes a fingerprint dataset.

        Args:
            fingerprint_file: path to a fingerprint .h5 file
        """
        self.fingerprints = self._load_fingerprints(fingerprint_file)

    def _load_fingerprints(self, fingerprint_file):
        """Loads fingerprint information."""
        f = h5py.File(fingerprint_file, 'r')
        
        fingerprint_data = f['fingerprints'][()]
        fingerprint_smiles = f['smiles'][()]
        
        # create smiles->idx mapping
        fingerprint_mapping = {}
        for i in range(len(fingerprint_smiles)):
            sm = fingerprint_smiles[i].decode('ascii')
            fingerprint_mapping[sm] = i
        
        fingerprints = {
            'fingerprint_data': fingerprint_data,
            'fingerprint_mapping': fingerprint_mapping,
            'fingerprint_smiles': fingerprint_smiles,
        }
        
        f.close()
        
        return fingerprints

    def for_smiles(self, smiles):
        """Return a Tensor of fingerprints for a list of smiles.
        
        Args:
            smiles: size N list of smiles strings (as str not bytes)
        """
        fp = np.zeros((len(smiles), self.fingerprints['fingerprint_data'].shape[1]))

        for i in range(len(smiles)):
            fp_idx = self.fingerprints['fingerprint_mapping'][smiles[i]]
            fp[i] = self.fingerprints['fingerprint_data'][fp_idx]

        return torch.Tensor(fp)


# class FragmentDataset2(Dataset):
#     '''
#     Utility class to work with the packed fragments.h5 format

#     (no fingerprints)
#     '''
    
#     def __init__(self, fragment_file, rec_typer, lig_typer, filter_rec=None, 
#         fdist_min=None, fdist_max=None, fmass_min=None, fmass_max=None, verbose=False):
#         '''
#         Initialize the fragment dataset
        
#         Params:
#         - fragment_file: path to fragments.h5
#         - rec_typer: function to map receptor rows to layer index
#         - lig_typer: function to map ligand rows to layer index
#         - filter_rec: list of receptor ids to use (or None to use all)

#         Filtering options:
#         - fdist_min: minimum fragment distance
#         - fdist_max: maximum fragment distance
#         - fmass_min: minimum fragment mass (Da)
#         - fmass_max: maximum fragment mass (Da)
#         '''
#         self.verbose = verbose

#         #  load receptor/fragment information
#         self.rec = self.load_rec(fragment_file, rec_typer)
#         self.frag = self.load_fragments(fragment_file, lig_typer)

#         # keep track of valid examples
#         valid_mask = np.ones(self.frag['frag_lookup'].shape[0]).astype(np.bool)
        
#         # filter by receptor id
#         if filter_rec is not None:
#             valid_rec = np.vectorize(lambda k: k[0].decode('ascii') in filter_rec)(self.frag['frag_lookup'])
#             valid_mask *= valid_rec
            
#         # filter by fragment distance
#         if fdist_min is not None:
#             valid_mask[self.frag['frag_dist'] < fdist_min] = 0
            
#         if fdist_max is not None:
#             valid_mask[self.frag['frag_dist'] > fdist_max] = 0
            
#         # filter by fragment mass
#         if fmass_min is not None:
#             valid_mask[self.frag['frag_mass'] < fmass_min] = 0

#         if fmass_max is not None:
#             valid_mask[self.frag['frag_mass'] > fmass_max] = 0

#         # convert to a list of indexes
#         self.valid_idx = np.where(valid_mask)[0]

#     def load_rec(self, fragment_file, rec_typer):
#         '''Load receptor information'''
#         f = h5py.File(fragment_file, 'r')
        
#         rec_coords = f['rec_coords'][()]
#         rec_types = f['rec_types'][()]
#         rec_lookup = f['rec_lookup'][()]
        
#         r = range(len(rec_types))
#         if self.verbose:
#             r = tqdm.tqdm(r, desc='Remap receptor atoms')

#         rec_remapped = np.zeros(len(rec_types)).astype(np.int32)
#         for i in r:
#             rec_remapped[i] = rec_typer(rec_types[i])

#         # create rec mapping
#         rec_mapping = {}
#         for i in range(len(rec_lookup)):
#             rec_mapping[rec_lookup[i][0].decode('ascii')] = i
        
#         rec = {
#             'rec_coords': rec_coords, 
#             'rec_types': rec_types,
#             'rec_remapped': rec_remapped,
#             'rec_lookup': rec_lookup,
#             'rec_mapping': rec_mapping
#         }
        
#         f.close()
        
#         return rec
        
#     def load_fragments(self, fragment_file, lig_typer):
#         '''Load fragment information'''

#         f = h5py.File(fragment_file, 'r')

#         frag_data = f['frag_data'][()]
#         frag_lookup = f['frag_lookup'][()]
#         frag_smiles = f['frag_smiles'][()]
#         frag_mass = f['frag_mass'][()]
#         frag_dist = f['frag_dist'][()]
        
#         # unpack frag data into separate structures
#         frag_coords = frag_data[:,:3].astype(np.float32)
#         frag_types = frag_data[:,3].astype(np.int32)
        
#         frag_remapped = np.vectorize(lig_typer)(frag_types)
        
#         # find and save connection point
#         r = range(len(frag_lookup))
#         if self.verbose:
#             r = tqdm.tqdm(r, desc='Frag connection point')

#         frag_conn = np.zeros((len(frag_lookup), 3))
#         for i in r:
#             _,f_start,f_end,_,_ = frag_lookup[i]
#             fdat = frag_data[f_start:f_end]
            
#             found = False
#             for j in range(len(fdat)):
#                 if fdat[j][3] == 0:
#                     frag_conn[i,:] = tuple(fdat[j])[:3]
#                     found = True
#                     break
                    
#             assert found, "missing fragment connection point at %d" % i
        
#         frag = {
#             'frag_coords': frag_coords,     # d_idx -> (x,y,z)
#             'frag_types': frag_types,       # d_idx -> (type)
#             'frag_remapped': frag_remapped, # d_idx -> (layer)
#             'frag_lookup': frag_lookup,     # f_idx -> (rec_id, fstart, fend, pstart, pend)
#             'frag_conn': frag_conn,         # f_idx -> (x,y,z)
#             'frag_smiles': frag_smiles,     # f_idx -> smiles
#             'frag_mass': frag_mass,         # f_idx -> mass
#             'frag_dist': frag_dist,         # f_idx -> dist
#         }
        
#         f.close()

#         return frag

#     def __len__(self):
#         '''returns the number of fragment examples'''
#         return self.valid_idx.shape[0]
    
#     def __getitem__(self, idx):
#         '''
#         retrieve the nth example
        
#         returns (f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, fingerprint, extra)
#         '''
#         # convert to fragment index
#         frag_idx = self.valid_idx[idx]
        
#         # lookup fragment
#         rec_id, f_start, f_end, p_start, p_end = self.frag['frag_lookup'][frag_idx]
#         smiles = self.frag['frag_smiles'][frag_idx]
#         conn = self.frag['frag_conn'][frag_idx]
        
#         # lookup receptor
#         rec_idx = self.rec['rec_mapping'][rec_id.decode('ascii')]
#         _, r_start, r_end = self.rec['rec_lookup'][rec_idx]
        
#         # fetch data
#         f_coords = self.frag['frag_coords'][f_start:f_end]
#         f_types = self.frag['frag_remapped'][f_start:f_end]
#         p_coords = self.frag['frag_coords'][p_start:p_end]
#         p_types = self.frag['frag_remapped'][p_start:p_end]
#         r_coords = self.rec['rec_coords'][r_start:r_end]
#         r_types = self.rec['rec_remapped'][r_start:r_end]
        
#         return f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, smiles