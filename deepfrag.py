import argparse
import functools
import os
import pathlib
import shutil
import time
from typing import Tuple
import zipfile

import requests
from tqdm.auto import tqdm
import h5py
import numpy as np
import rdkit.Chem.AllChem as Chem
import torch
import prody

from leadopt.model_conf import LeadoptModel, REC_TYPER, LIG_TYPER, DIST_FN
from leadopt import util, grid_util


USER_DIR = './.store'
PDB_CACHE = 'pdb_cache'

MODEL_DOWNLOAD = 'https://durrantlab.pitt.edu/apps/deepfrag/files/final_model_v2.zip'
FINGERPRINTS_DOWNLOAD = 'https://durrantlab.pitt.edu/apps/deepfrag/files/fingerprints.h5'

RCSB_DOWNLOAD = 'https://files.rcsb.org/download/%s.pdb1'

VERSION = "1.0.2"

def download_remote(url, path, compression=None):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        print(f'Can\'t access {url}')

    file_size = int(r.headers.get('Content-Length', 0))

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, 'read', total=file_size, desc='Downloading') as r_raw:
        with path.open('wb') as f:
            shutil.copyfileobj(r_raw, f)

    if compression is not None:
        shutil.move(str(path), str(path) + '.tmp')
        shutil.unpack_archive(str(path) + '.tmp', str(path), format=compression)


def get_deepfrag_user_dir() -> pathlib.Path:
    user_dir = pathlib.Path(os.path.realpath(__file__)).parent / USER_DIR
    os.makedirs(str(user_dir), exist_ok=True)
    return user_dir


def get_model_path():
    return get_deepfrag_user_dir() / 'model'


def get_fingerprints_path():
    return get_deepfrag_user_dir() / 'fingerprints.h5'


def ensure_cli_data():
    model_path = get_model_path()
    fingerprints_path = get_fingerprints_path()

    if not os.path.exists(str(model_path)):
        r = input('Pre-trained DeepFrag model not found, download it now? (5.8 MB) [Y/n]: ')
        if r.lower() == 'n':
            print('Exiting...')
            exit(-1)

        print(f'Saving to {model_path}...')
        download_remote(MODEL_DOWNLOAD, model_path, compression='zip')

    if not os.path.exists(str(fingerprints_path)):
        r = input('Fingerprint library not found, download it now? (11 MB) [Y/n]: ')
        if r.lower() == 'n':
            print('Exiting...')
            exit(-1)

        print(f'Saving to {fingerprints_path}...')
        download_remote(FINGERPRINTS_DOWNLOAD, fingerprints_path, compression=None)


def download_pdb(pdb_id, path):
    download_remote(RCSB_DOWNLOAD % pdb_id, path, compression=None)


def load_pdb(pdb_id, resnum):
    pdb_id = pdb_id.upper()
    assert all([x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for x in pdb_id])

    # Check pdb cache
    pdb_dir = get_deepfrag_user_dir() / PDB_CACHE / pdb_id

    complex_path = pdb_dir / 'complex.pdb'
    rec_path = pdb_dir / 'receptor.pdb'
    lig_path = pdb_dir / 'ligand.pdb'

    os.makedirs(str(pdb_dir), exist_ok=True)

    if not os.path.exists(complex_path):
        download_pdb(pdb_id, complex_path)

    m = prody.parsePDB(str(complex_path))
    rec = m.select('not (nucleic or hetatm) and not water')
    lig = m.select('resnum %d' % resnum)

    if lig is None:
        print('[!] Error could not find ligand with resnum: %d' % resnum)
        exit(-1)

    prody.writePDB(str(rec_path), rec)
    prody.writePDB(str(lig_path), lig)

    return (str(rec_path), str(lig_path))


def get_structure_paths(args) -> Tuple[str, str]:
    """Get structure paths specified by the command line args.
    Returns (rec_path, lig_path)
    """
    if args.receptor is not None and args.ligand is not None:
        return (args.receptor, args.ligand)
    elif args.pdb is not None and args.resnum is not None:
        return load_pdb(args.pdb, args.resnum)
    else:
        raise NotImplementedError()


def preprocess_ligand(lig, conn, rvec):
    """
    Remove the fragment from lig connected via the atom at conn and containing
    the atom at rvec.
    """
    # Generate all fragments.
    frags = util.generate_fragments(lig)

    for parent, frag in frags:
        cidx = [a for a in frag.GetAtoms() if a.GetAtomicNum() == 0][0].GetIdx()
        vec = frag.GetConformer().GetAtomPosition(cidx)
        c_vec = np.array([vec.x, vec.y, vec.z])

        # Check connection point.
        if np.linalg.norm(c_vec - conn) < 1e-3:
            # Check removal point.
            frag_pos = frag.GetConformer().GetPositions()
            min_dist = np.min(np.sum((frag_pos - rvec) ** 2, axis=1))

            if min_dist < 1e-3:
                # Found fragment.
                print('[*] Removing fragment with %d atoms (%s)' % (
                    frag_pos.shape[0] - 1, Chem.MolToSmiles(frag, False)))

                return parent

    print('[!] Could not find a suitable fragment to remove.')
    exit(-1)


def lookup_atom_name(lig_path, name):
    """Try to look up an atom by name. Returns the coordinate of the atom if
    found."""
    p = prody.parsePDB(lig_path)
    p = p.select('name %s' % name)
    if p is None:
        print('[!] Error: no atom with name "%s" in ligand' % name)
        exit(-1)
    elif len(p) > 1:
        print('[!] Error: multiple atoms with name "%s" in ligand' % name)
        exit(-1)
    return p.getCoords()[0]


def get_structures(args):
    rec_path, lig_path = get_structure_paths(args)

    print(f'[*] Loading receptor: {rec_path} ... ', end='')
    rec_coords, rec_types = util.load_receptor_ob(rec_path)
    print('done.')

    print(f'[*] Loading ligand: {lig_path} ... ', end='')
    lig = Chem.MolFromPDBFile(lig_path)
    print('done.')

    conn = None
    if args.cx is not None and args.cy is not None and args.cz is not None:
        conn = np.array([float(args.cx), float(args.cy), float(args.cz)])
    elif args.cname is not None:
        conn = lookup_atom_name(lig_path, args.cname)
    else:
        raise NotImplementedError()

    rvec = None
    if args.rx is not None and args.ry is not None and args.rz is not None:
        rvec = np.array([float(args.rx), float(args.ry), float(args.rz)])
    elif args.rname is not None:
        rvec = lookup_atom_name(lig_path, args.rname)
    else:
        pass

    if rvec is not None:
        lig = preprocess_ligand(lig, conn, rvec)

    parent_coords = util.get_coords(lig)
    parent_types = np.array(util.get_types(lig)).reshape((-1,1))

    return (rec_coords, rec_types, parent_coords, parent_types, conn, lig)


def get_model(args, device):
    """Load a pre-trained DeepFrag model."""
    print('[*] Loading model ... ', end='')
    model = LeadoptModel.load(str(get_model_path() / 'final_model'), device=('cuda' if device == 'gpu' else device))
    print('done.')
    return model


def get_fingerprints(args):
    """Load the fingerprint library.
    Returns (smiles, fingerprints).
    """
    f_smiles = None
    f_fingerprints = None
    print('[*] Loading fingerprint library ... ', end='')
    with h5py.File(str(get_fingerprints_path()), 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(np.float)
    print('done.')

    return (f_smiles, f_fingerprints)


def get_target_device(args) -> str:
    """Infer the target device or use the argument overrides."""
    device = 'gpu' if torch.cuda.device_count() > 0 else 'cpu'

    if args.cpu:
        if device == 'gpu':
            print('[*] Warning: GPU is available but running on CPU due to --cpu flag')
        device = 'cpu'
    elif args.gpu:
        if device == 'cpu':
            print('[*] Error: No CUDA-enabled GPU was found. Exiting due to --gpu flag. You can run on the CPU instead with the --cpu flag.')
            exit(-1)
        device = 'gpu'

    print('[*] Running on device: %s' % device)

    return device


def generate_grids(args, model_args, rec_coords, rec_types, parent_coords, parent_types, conn, device):
    start = time.time()

    print('[*] Generating grids ... ', end='', flush=True)
    batch = grid_util.get_raw_batch(
        rec_coords, rec_types, parent_coords, parent_types,
        rec_typer=REC_TYPER[model_args['rec_typer']],
        lig_typer=LIG_TYPER[model_args['lig_typer']],
        conn=conn,
        num_samples=args.num_grids,
        width=model_args['grid_width'],
        res=model_args['grid_res'],
        point_radius=model_args['point_radius'],
        point_type=model_args['point_type'],
        acc_type=model_args['acc_type'],
        cpu=(device == 'cpu')
    )
    print('done.')
    end = time.time()
    print(f'[*] Generated grids in {end-start:.3f} seconds.')

    return batch


def get_predictions(model, batch, f_smiles, f_fingerprints):
    start = time.time()
    pred = model.predict(torch.tensor(batch).float()).cpu().numpy()
    end = time.time()
    print(f'[*] Generated prediction in {end-start} seconds.')

    avg_fp = np.mean(pred, axis=0)
    dist_fn = DIST_FN[model._args['dist_fn']]

    # The distance functions are implemented in pytorch so we need to convert our
    # numpy arrays to a torch Tensor.
    dist = 1 - dist_fn(
        torch.tensor(avg_fp).unsqueeze(0),
        torch.tensor(f_fingerprints))

    # Pair smiles strings and distances.
    dist = list(dist.numpy())
    scores = list(zip(f_smiles, dist))
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    scores = [(a.decode('ascii'), b) for a,b in scores]

    return scores


def gen_output(args, scores):
    if args.out is None:
        # Write results to stdout.
        print('%4s %8s %s' % ('#', 'Score', 'SMILES'))
        for i in range(len(scores)):
            smi, score = scores[i]
            print('%4d %8f %s' % (i+1, score, smi))
    else:
        # Write csv output.
        csv = 'Rank,SMILES,Score\n'
        for i in range(len(scores)):
            smi, score = scores[i]
            csv += '%d,%s,%f\n' % (
                i+1, smi, score
            )

        open(args.out, 'w').write(csv)
        print('[*] Wrote output to %s' % args.out)


def fuse(lig, frag):
    merged = Chem.RWMol(Chem.CombineMols(lig, frag))

    conn_atoms = [a.GetIdx() for a in merged.GetAtoms() if a.GetAtomicNum() == 0]
    neighbors = [merged.GetAtomWithIdx(x).GetNeighbors()[0].GetIdx() for x in conn_atoms]

    bond = merged.AddBond(neighbors[0], neighbors[1], Chem.rdchem.BondType.SINGLE)

    merged.RemoveAtom([a.GetIdx() for a in merged.GetAtoms() if a.GetAtomicNum() == 0][0])
    merged.RemoveAtom([a.GetIdx() for a in merged.GetAtoms() if a.GetAtomicNum() == 0][0])

    Chem.SanitizeMol(merged)

    return merged


def fuse_fragments(lig, conn, scores):
    new_sc = []
    for smi, score in scores:
        try:
            frag = Chem.MolFromSmiles(smi)
            fused = fuse(Chem.Mol(lig), frag)
            new_sc.append((Chem.MolToSmiles(fused, False), score))
        except:
            print('[*] Error: couldn\'t process mol.')
            new_sc.append(('<err>', score))

    return new_sc


def run(args):
    device = get_target_device(args)

    model = get_model(args, device)
    f_smiles, f_fingerprints = get_fingerprints(args)

    rec_coords, rec_types, parent_coords, parent_types, conn, lig = get_structures(args)

    batch = generate_grids(args, model._args, rec_coords, rec_types,
        parent_coords, parent_types, conn, device)

    scores = get_predictions(model, batch, f_smiles, f_fingerprints)

    if args.top_k != -1:
        scores = scores[:args.top_k]

    if args.full:
        scores = fuse_fragments(lig, conn, scores)

    gen_output(args, scores)


def main():
    global VERSION

    print("\nDeepFrag " + VERSION)
    print("\nIf you use DeepFrag in your research, please cite:\n")
    print("Green, H., Koes, D. R., & Durrant, J. D. (2021). DeepFrag: a deep convolutional")
    print("neural network for fragment-based lead optimization. Chemical Science.\n")


    ensure_cli_data()

    parser = argparse.ArgumentParser()

    # Structure
    parser.add_argument('--receptor', help='Path to receptor structure.')
    parser.add_argument('--ligand', help='Path to ligand structure.')
    parser.add_argument('--pdb', help='PDB ID to download.')
    parser.add_argument('--resnum', type=int, help='Residue number of ligand.')

    # Connection point
    parser.add_argument('--cx', type=int, help='Connection point x coordinate.')
    parser.add_argument('--cy', type=int, help='Connection point y coordinate.')
    parser.add_argument('--cz', type=int, help='Connection point z coordinate.')
    parser.add_argument('--cname', type=str, help='Connection point atom name.')

    # Removal point
    parser.add_argument('--rx', type=int, help='Removal point x coordinate.')
    parser.add_argument('--ry', type=int, help='Removal point y coordinate.')
    parser.add_argument('--rz', type=int, help='Removal point z coordinate.')
    parser.add_argument('--rname', type=str, help='Removal point atom name.')

    # Misc
    parser.add_argument('--full', action='store_true', default=False,
        help='Print the full (fused) ligand structure.')
    parser.add_argument('--num_grids', type=int, default=4,
        help='Number of grid rotations.')
    parser.add_argument('--top_k', type=int, default=25,
        help='Number of results to show. Set to -1 to show all.')
    parser.add_argument('--out', type=str,
        help='Path to output CSV file.')
    parser.add_argument('--cpu', action='store_true', default=False,
        help='Use the CPU for grid generation and predictions.')
    parser.add_argument('--gpu', action='store_true', default=False,
        help='Use a (CUDA-capable) GPU for grid generation and predictions.')

    args = parser.parse_args()

    groupings = [
        ([('receptor', 'ligand'), ('pdb', 'resnum')], True),
        ([('cx', 'cy', 'cz'), ('cname',)], True),
        ([('rx', 'ry', 'rz'), ('rname',)], False),
        ([('cpu',), ('gpu',)], False)
    ]

    for grp, req in groupings:
        partial = []
        complete = 0

        for subset in grp:
            res = [not (getattr(args, name) in [None, False]) for name in subset]
            partial.append(any(res) and not all(res))
            complete += int(all(res))

        if any(partial) or complete > 1 or (complete != 1 and req):
            # Invalid arg combination.
            print('Invalid arguments, must specify exactly one of the following combinations:')
            for subset in grp:
                print('\t%s' % ', '.join(['--' + x for x in subset]))
            exit(-1)

    run(args)


if __name__=='__main__':
    main()
