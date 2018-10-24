import openbabel
import numpy as np

import grid

class Mol(object):
    '''
    Minimal representation of a ligand/fragment.
    '''

    def __init__(self):
        self.atoms = []
        self.conn = []
        self.frag = []
    
    def to_string(self):
        '''
        Convert to csv-like string format:

        n
        atom_0.x,atom_0.y,atom_0.z,atom_0.t
        atom_1.x,atom_1.y,atom_1.z,atom_1.t
        ...
        atom_n-1.x,atom_n-1.y,atom_n-1.z,atom_n-1.t
        c
        conn_0.x,conn_0.y,conn_0.z,conn_0.t
        conn_1.x,conn_1.y,conn_1.z,conn_1.t
        ...
        conn_c-1.x,conn_c-1.y,conn_c-1.z,conn_c-1.t
        m
        frag_0.x,frag_0.y,frag_0.z,frag_0.t
        frag_1.x,frag_1.y,frag_1.z,frag_1.t
        ...
        frag_m-1.x,frag_m-1.y,frag_m-1.z,frag_m-1.t
        '''
        s = ''

        s += str(len(self.atoms)) + '\n'
        for a in self.atoms:
            s += ','.join(map(str, a)) + '\n'

        s += str(len(self.conn)) + '\n'
        for c in self.conn:
            s += ','.join(map(str, c)) + '\n'

        s += str(len(self.frag)) + '\n'
        for f in self.frag:
            s += ','.join(map(str, f)) + '\n'

        return s

    def center(self):
        '''
        Returns the average center point of all the atoms
        '''
        p = []

        for a in (self.atoms + self.conn + self.frag):
            p.append(a[:-1])

        return np.mean(p, axis=0)
    
    @staticmethod
    def from_pybel(pybel_molecule, fragment=None):
        '''
        Create a Mol from a given pybel.Molecule. Optionally, provide a list of atoms
        to consider as a separate fragment.

        Note: atoms are indexed starting from 1
        '''
        m = Mol()

        for a in pybel_molecule.atoms:
            # get atomic number
            n = a.atomicnum

            # get coordinates
            [x,y,z] = a.coords

            if fragment is not None and a.idx in fragment:
                m.frag.append((x,y,z,n))
            else:
                # check if this atom connects to a fragment atom
                neighbors = [h.GetIdx() for h in openbabel.OBAtomAtomIter(a.OBAtom)]

                is_connector = False
                for ne in neighbors:
                    if fragment is not None and ne in fragment:
                        is_connector = True
                        break

                if is_connector:
                    m.conn.append((x,y,z,n))
                else:
                    m.atoms.append((x,y,z,n))

        return m

    @staticmethod
    def writefile(filename, mols):
        '''
        Write a list of mols to a file
        '''
        with open(filename, 'w') as f:
            for m in mols:
                f.write(m.to_string())

    @staticmethod
    def readfile(filename):
        '''
        Read a list of Mols from a file
        '''
        dat = open(filename).read().strip().split('\n')

        if dat[0] == '':
            return []

        mols = []

        i = 0
        while i < len(dat):
            m = Mol()

            # read number of atoms
            num_atoms = int(dat[i])

            for atom_line in dat[i+1:i+1+num_atoms]:
                [x,y,z,t] = atom_line.split(',')

                m.atoms.append((float(x), float(y), float(z), int(t)))

            i += num_atoms + 1

            # read number of connector atoms
            num_conn = int(dat[i])

            for conn_line in dat[i+1:i+1+num_conn]:
                [x,y,z,t] = conn_line.split(',')

                m.conn.append((float(x), float(y), float(z), int(t)))

            i += num_conn + 1

            # read number of fragment atoms
            num_frag = int(dat[i])

            for frag_line in dat[i+1:i+1+num_frag]:
                [x,y,z,t] = frag_line.split(',')

                m.frag.append((float(x), float(y), float(z), int(t)))

            i += num_frag + 1

            mols.append(m)

        return mols
