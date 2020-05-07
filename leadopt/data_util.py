'''
data_util.py

contains utility code for reading packed training data
'''
import os

from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py


# Default atomic numbers to use as grid layers
DEFAULT_TYPES = [6,7,8,9,15,16,17,35,53]


def remap_atoms(atom_types):
    '''
    Returns a function that maps an atomic number to layer index.

    Params:
    - atom_types: which atom types to use as layers
    '''
    atom_mapping = {atom_types[i]:i for i in range(len(atom_types))}
    
    def f(x):
        if x in atom_mapping:
            return atom_mapping[x]
        else:
            return -1
    
    return f


class FragmentDataset(Dataset):
    '''
    Utility class to work with the packed fragments.h5 format
    '''
    
    def __init__(self, fragment_file, fingerprint_file, filter_rec=None, atom_types=DEFAULT_TYPES, fdist_min=None, fdist_max=None, fmass_min=None, fmass_max=None):
        '''
        Initialize the fragment dataset
        
        Params:
        - fragment_file: path to fragments.h5
        - fingerprint_file: path to fingerprints.h5
        - filter_rec: list of receptor ids to use (or None to use all)
        - atom_types: which atom types to use as layers

        Filtering options:
        - fdist_min: minimum fragment distance
        - fdist_max: maximum fragment distance
        - fmass_min: minimum fragment mass (Da)
        - fmass_max: maximum fragment mass (Da)
        '''
        #  load receptor/fragment information
        self.rec = self.load_rec(fragment_file, atom_types)
        self.frag = self.load_fragments(fragment_file, atom_types)

        # load fingerprint information
        self.fingerprints = self.load_fingerprints(fingerprint_file)
        
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
        self.valid_idx = np.where(valid_mask)[0]

        # compute frequency metrics over valid fragments
        # (valid idx -> (smiles count))
        self.freq = self.compute_freq()

        self.valid_fingerprints = self.compute_valid_fingerprints()

    def load_rec(self, fragment_file, atom_types):
        '''Load receptor information'''
        f = h5py.File(fragment_file, 'r')
        
        rec_data = f['rec_data'][()]
        rec_lookup = f['rec_lookup'][()]
        
        # unpack rec data into separate structures
        rec_coords = rec_data[:,:3].astype(np.float32)
        rec_types = rec_data[:,3].reshape(-1,1).astype(np.int32)
        
        rec_remapped = np.vectorize(remap_atoms(atom_types))(rec_types)
        
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
        
    def load_fragments(self, fragment_file, atom_types):
        '''Load fragment information'''

        f = h5py.File(fragment_file, 'r')

        frag_data = f['frag_data'][()]
        frag_lookup = f['frag_lookup'][()]
        frag_smiles = f['frag_smiles'][()]
        frag_mass = f['frag_mass'][()]
        frag_dist = f['frag_dist'][()]
        
        # unpack frag data into separate structures
        frag_coords = frag_data[:,:3].astype(np.float32)
        frag_types = frag_data[:,3].reshape(-1,1).astype(np.int32)
        
        frag_remapped = np.vectorize(remap_atoms(atom_types))(frag_types)
        
        # find and save connection point
        frag_conn = np.zeros((len(frag_lookup), 3))
        for i in range(len(frag_lookup)):
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
    
    def load_fingerprints(self, fingerprint_file):
        '''load fingerprint information'''
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

    def compute_freq(self):
        '''compute fragment frequencies'''
        all_smiles = self.frag['frag_smiles']
        valid_smiles = all_smiles[self.valid_idx]

        smiles_freq = {}
        for i in range(len(valid_smiles)):
            sm = valid_smiles[i].decode('ascii')
            if not sm in smiles_freq:
                smiles_freq[sm] = 0
            smiles_freq[sm] += 1

        freq = np.zeros(len(valid_smiles))
        for i in range(len(valid_smiles)):
            freq[i] = smiles_freq[valid_smiles[i].decode('ascii')]

        return freq

    def compute_valid_fingerprints(self):
        '''compute a list of valid fingerprint indexes'''
        valid_sm = self.frag['frag_smiles'][self.valid_idx]
        valid_sm = list(set(list(valid_sm))) # unique

        valid_idx = []
        for sm in valid_sm:
            valid_idx.append(self.fingerprints['fingerprint_mapping'][sm.decode('ascii')])
        valid_idx = sorted(valid_idx)

        return valid_idx

    def normalize_fingerprints(self, std, mean):
        '''normalize fingerprints with a given std and mean'''
        self.fingerprints['fingerprint_data'] -= mean
        self.fingerprints['fingerprint_data'] /= std
        
    def __len__(self):
        '''returns the number of fragment examples'''
        return self.valid_idx.shape[0]
    
    def __getitem__(self, idx):
        '''
        retrieve the nth example
        
        returns (f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, fingerprint, extra)
        '''
        # convert to fragment index
        frag_idx = self.valid_idx[idx]
        
        # lookup fragment
        rec_id, f_start, f_end, p_start, p_end = self.frag['frag_lookup'][frag_idx]
        smiles = self.frag['frag_smiles'][frag_idx]
        conn = self.frag['frag_conn'][frag_idx]
        
        # lookup fingerprint
        fingerprint = self.fingerprints['fingerprint_data'][
            self.fingerprints['fingerprint_mapping'][smiles.decode('ascii')]
        ]
        
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

        extra = {
            'freq': self.freq[idx]
        }
        
        return f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, fingerprint, extra
    