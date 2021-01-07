# Copyright 2021 Jacob Durrant

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


# MOAD csv parsing utility

class Family(object):
    def __init__(self):
        self.targets = []

    def __repr__(self):
        return 'F(%d)' % len(self.targets)

class Protein(object):
    def __init__(self, pdb_id):
        # (chain, smi)
        self.pdb_id = pdb_id.upper()
        self.ligands = []

    def __repr__(self):
        return '%s(%d)' % (self.pdb_id, len(self.ligands))

def parse_moad(csv):
    csv_dat = open(csv, 'r').read().strip().split('\n')
    csv_dat = [x.split(',') for x in csv_dat]

    families = []

    curr_f = None
    curr_t = None

    for line in csv_dat:
        if line[0] != '':
            # new class
            continue
        elif line[1] != '':
            # new family
            if curr_t != None:
                curr_f.targets.append(curr_t)
            if curr_f != None:
                families.append(curr_f)
            curr_f = Family()
            curr_t = Protein(line[2])
        elif line[2] != '':
            # new target
            if curr_t != None:
                curr_f.targets.append(curr_t)
            curr_t = Protein(line[2])
        elif line[3] != '':
            # new ligand
            if line[4] != 'valid':
                continue
            curr_t.ligands.append((line[3], line[9]))

    curr_f.targets.append(curr_t)
    families.append(curr_f)

    by_target = {}
    for f in families:
        for t in f.targets:
            by_target[t.pdb_id] = t

    return families, by_target
