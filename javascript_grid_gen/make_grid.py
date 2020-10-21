"""
Create a protein/ligand grid. Can compile to JavaScript using Transcrypt.
"""

from gridder.util import load_receptor, load_ligand, mol_to_points, get_connection_point
from gridder.grid_util import get_raw_batch_one_channel
from gridder.fake_rdkit import Point

# "" "?
# fr om gridder.example_data import receptor_pdb, ligand_sdf
# ?" ""

# Data Format
#
# Receptor data:
# - rec_lookup:     [id][start][end]
# - rec_coords:     [x][y][z]
# - rec_types:      [num][is_hacc][is_hdon][is_aro][pcharge]
#
# Fragment data:
# - frag_lookup:    [id][fstart][fend][pstart][pend]
# - frag_coords:    [x][y][z]
# - frag_types:     [num]
# - frag_smiles:    [smiles]
# - frag_mass:      [mass]
# - frag_dist:      [dist]

# "" "?
# def ge t_test_data():
#     return receptor_pdb, ligand_sdf
# ?" ""


def pre_grid_gen(receptor: str, ligand: str, grid_center: list):
    # load ligand and receptor

    rec = load_receptor(receptor)

    # If SDF
    # lig, frags = load_ligand(ligand)  # List of tuples, (parent, frag)

    # If PDB
    # import pdb;pdb.set_trace()
    lig = load_receptor(ligand)  # load_receptor because only receptor was originally in PDB format.
    # lig, frags = None, [(l, None)]  # to get pdb to work

    # compute shared receptor coords and layers
    rec_coords, rec_layers = mol_to_points(rec, None, note_sulfur=True)

    # frag = frags[0][1]

    # use ligand directly (already fragmented)
    # compute parent coords and layers
    parent_coords, parent_layers = mol_to_points(lig, None, note_sulfur=False)

    # find connection point
    # __pragma__ ('skip')
    # try:
    #     conn = get_connection_point(frag)
    # except:
    # If runing under python with fake_rdkit
    conn = Point(grid_center)
    # __pragma__ ('noskip')

    """?
    conn = Point(grid_center)
    ?"""

    return rec_coords, rec_layers, parent_coords, parent_layers, conn


def make_grid_given_channel(
    rec_coords, rec_layers, parent_coords, parent_layers, conn, channel: int
):
    # Note: Setting it up this way to enable parallel processing in future, if
    # necessary.
    return get_raw_batch_one_channel(
        rec_coords,
        rec_layers,
        parent_coords,
        parent_layers,
        conn,
        1,
        24,
        0.75,
        channel,
    )

def sum_channel_grids(grids):
    # summed_grid = [0] * len(grids[0])  # Doesn't work in transcrypt
    summed_grid = [0 for i in range(len(grids[0]))]
    for i in range(len(summed_grid)):
        for grid in grids:
            summed_grid[i] = summed_grid[i] + grid[i]
    return summed_grid



if __name__ == "__main__":
    # __pragma__ ('skip')

    # Below demos how to use it, but really these should all be called
    # directly from javascript...

    receptor = "11gs/11gs_protein.pdb"
    # ligand = "11gs/11gs_ligand.minus-grid0-frag.sdf"
    ligand = "11gs/11gs_ligand.minus-grid0-frag.sdf.pdb"
    grid_center = [14.62, 9.944, 24.471]

    rec_coords, rec_layers, parent_coords, parent_layers, conn = pre_grid_gen(
        receptor, ligand, grid_center
    )

    grids = [
        make_grid_given_channel(
            rec_coords, rec_layers, parent_coords, parent_layers, conn, i
        )
        for i in range(9)
    ]

    # You must sum the grids.
    summed_grid = sum_channel_grids(grids)

    print(summed_grid)

    # __pragma__ ('noskip')

    # "" "?
    # # make _grid(
    # #     receptor_pdb,
    # #     ligand_sdf,
    # #     [14.62, 9.944, 24.471],
    # # )
    # ?" ""
