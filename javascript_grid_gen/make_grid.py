"""
Create a protein/ligand grid. Can compile to JavaScript using Transcrypt.
"""

from gridder.util import load_receptor, load_ligand, mol_to_points, get_connection_point
from gridder.grid_util import get_raw_batch
from gridder.fake_rdkit import Point

# __pragma__ ('skip')
import json

# __pragma__ ('noskip')

"""?
from gridder.example_data import receptor_pdb, ligand_sdf
?"""

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

"""?
def get_test_data():
    return receptor_pdb, ligand_sdf
?"""


def make_grid(receptor: str, ligand: str, grid_center: list) -> None:
    """Makes a grid from the receptor or ligand.

    :param receptor:    The receptor. A file name if running from the command
                        line, or the PDB-formated contents if running in the
                        browser.
    :type receptor:     str
    :param ligand:      The ligand. A file name if running from the command
                        line, or the SDF-formated contents if running in the
                        browser.
    :type ligand:       str
    :param grid_center: The center of the grid. In the browser, it will be
                        identified using 3Dmol.js.
    :type grid_center: List
    """

    # load ligand and receptor

    rec = load_receptor(receptor)

    # If SDF
    lig, frags = load_ligand(ligand)  # List of tuples, (parent, frag)

    # If PDB
    # l = load_receptor(ligand)
    # lig, frags = None, [(l, None)]  # to get pdb to work

    # compute shared receptor coords and layers
    rec_coords, rec_layers = mol_to_points(rec, None, note_sulfur=True)

    # Only keep the first fragment. A JDD addition.
    # frags = frags[:1]
    frags = [frags[0]]

    for parent, frag in frags:

        # use ligand directly (already fragmented)
        # compute parent coords and layers
        parent_coords, parent_layers = mol_to_points(lig, None, note_sulfur=False)

        # find connection point
        # __pragma__ ('skip')
        try:
            conn = get_connection_point(frag)
        except:
            # If runing under python with fake_rdkit
            conn = Point(grid_center)
        # __pragma__ ('noskip')

        """?
        conn = Point(grid_center)
        ?"""

        # generate batch
        grid = get_raw_batch(
            rec_coords,
            rec_layers,
            parent_coords,
            parent_layers,
            conn,
            1,
            24,  # width=
            0.75,  # res=
        )

        # __pragma__ ('skip')
        return json.dumps(grid)
        # __pragma__ ('noskip')

        """?
        # print(str(grid))
        return grid
        ?"""


if __name__ == "__main__":
    # __pragma__ ('skip')
    grid = (
        make_grid(
            # "./1b6l/1b6l_protein.pdb",
            "11gs/11gs_protein.pdb",
            # "./1b6l/1b6l_ligand.sdf",
            "11gs/11gs_ligand.minus-grid0-frag.sdf",
            # [0.512000, 3.311000, 12.006000],
            [14.62, 9.944, 24.471],
        )
    )
    open('./11gs/mol_gridify2.json', 'w').write(grid)
    # __pragma__ ('noskip')

    """?
    # make_grid(
    #     receptor_pdb,
    #     ligand_sdf,
    #     [14.62, 9.944, 24.471],
    # )
    ?"""
