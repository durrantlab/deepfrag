# Pure python rdkit substitute

import re

element_names_with_two_letters = [
    "BR",
    "CL",
    "BI",
    "AS",
    "AG",
    "LI",
    "MG",
    "RH",
    "ZN",
    "MN",
]

element_to_atomic_num = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
}


class Point:
    """Represents a 3D coordinate."""

    def __init__(self, coor: list) -> None:
        """Constructor.

        :param coor: The 3D coordinate.
        :type coor: list
        """
        self.x = coor[0]
        self.y = coor[1]
        self.z = coor[2]

    def __getitem__(self, key: int) -> float:
        """Gets one of the coordinate values.

        :param key: The coordinate index. 0 = x, 1 = y, 2 = z.
        :type key: int
        :return: The coordinate value.
        :rtype: float
        """
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        if key == 2:
            return self.z

        assert False
        return 0


class Atom:
    """Mimics RDKit atom object."""

    def __init__(self, coor: list, name: str, element: str, resname: str) -> None:
        """Constructor.

        :param coor: The coordinate.
        :type coor: list
        :param name: The name of the atom.
        :type name: str
        :param element: The element symbol of the atom.
        :type element: str
        :param resname: The resname the atom belongs to.
        :type resname: str
        """

        self.coor = Point(coor)
        self.name = name
        self.element = element
        self.resname = resname

    def GetAtomicNum(self) -> int:
        """Gets the atomic number of atom.

        :return: The number.
        :rtype: int
        """

        return element_to_atomic_num[self.element]


class Mol:
    """Represents an RDKit Mol object."""

    def __init__(self):
        """Constructor."""

        self.atoms = []

    def add_atom(self, coor: list, name: str, element: str, resname: str) -> None:
        """Adds an atom to this Mol object.

        :param coor: The 3D coordinate.
        :type coor: list
        :param name: The name.
        :type name: str
        :param element: The element symbol.
        :type element: str
        :param resname: The resname to which the atom belongs.
        :type resname: str
        """

        self.atoms.append(Atom(coor, name, element, resname))

    def GetAtomWithIdx(self, i: int) -> Atom:
        """Get the atom with the specified index.

        :param i: The index.
        :type i: int
        :return: The atom.
        :rtype: Atom
        """
        return self.atoms[i]

    def GetNumAtoms(self) -> int:
        """Get the number of atoms in this Mol object.

        :return: The number.
        :rtype: int
        """

        return len(self.atoms)

    def GetConformer(self):
        """Returns this Mol object. Just for compatibility.

        :return: This Mol object.
        :rtype: Mol
        """

        return self

    def GetAtomPosition(self, i: int) -> list:
        """Get the 3D coordinate of an atom.

        :param i: The index of the atom.
        :type i: int
        :return: The 3D coordinate.
        :rtype: list
        """
        return self.atoms[i].coor


class MolIterator:
    """For iterating through molecules."""

    def __init__(self, mol: Mol) -> None:
        """Constructor.

        :param mol: The molecule. Really just iterating through one molecule.
            For compatibilty.
        :type mol: Mol
        """
        self.mols = [mol]

    def __iter__(self):
        return self

    def __next__(self) -> Mol:
        """Gets the next molecule.

        :raises StopIteration: Once done.
        :return: The next Mol.
        :rtype: Mol
        """

        if len(self.mols) == 0:
            raise StopIteration
        return self.mols.pop()


class Chem:
    """Mimics the RDKit.Chem object."""

    @staticmethod
    def MolFromPDBFile(filetxt: str, sanitize: bool) -> Mol:
        """Get a Mol object from PDB.

        :param filetxt: The file name (if command line) or the contents of the
            file (if Javascript).
        :type filetxt: str
        :param sanitize: Not used. Just for compatibility.
        :type sanitize: bool
        :return: The Mol object.
        :rtype: Mol
        """

        mol = Mol()

        # __pragma__ ('skip')
        try:
            with open(filetxt, "r") as f:
                lines_src = f.readlines()
        except:
            # running python with fake_rdkit
            lines_src = filetxt.split("\n")
        # __pragma__ ('noskip')

        """?
        lines_src = filetxt.split("\n")
        ?"""

        lines = [l for l in lines_src if l.startswith("ATOM") or l.startswith("HETATM")]

        coors = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines]
        names = [l[11:16].strip() for l in lines]
        elements_prep = [n.upper().strip() for n in names]
        for i, e in enumerate(elements_prep):
            for num in "0123456789":
                e = e.replace(num, "")
            elements_prep[i] = e
        elements_prep = [e[:2] for e in elements_prep]

        elements = []
        for e in elements_prep:
            elements.append(Chem.name_to_element(e))

        resnames = [l[16:21].strip() for l in lines]

        for i in range(len(coors)):
            mol.add_atom(coors[i], names[i], elements[i], resnames[i])

        return mol

    @staticmethod
    def name_to_element(name: str) -> str:
        """Get the element symbol based on the atom name.

        :param name: The atom name.
        :type name: str
        :return: The element symbol.
        :rtype: str
        """

        if name in element_names_with_two_letters:
            return name
        else:
            return name[:1]

    @staticmethod
    def SDMolSupplier(filetxt, sanitize=False):
        """Get a Mol object from SDF.

        :param filetxt: The file name (if command line) or the contents of the
            file (if Javascript).
        :type filetxt: str
        :param sanitize: Not used. Just for compatibility.
        :type sanitize: bool
        :return: The Mol object.
        :rtype: Mol
        """

        mol = Mol()

        # __pragma__ ('skip')
        try:
            with open(filetxt, "r") as f:
                txt = f.read()
        except:
            # running python with fake_rdkit
            txt = filetxt
        # __pragma__ ('noskip')

        """?
        txt = filetxt
        ?"""

        atoms = re.findall(
            "^ *?[\-0-9]+?\.[\-0-9]+? *?[\-0-9]+?\.[\-0-9]+? *?[\-0-9]+?\.[\-0-9]+? *?[a-zA-Z]+? *?[\-0-9]+? *?[\-0-9]+? *?[\-0-9]+? *?[\-0-9]+? *?[\-0-9]+? *?[\-0-9]+? *?$",
            txt,
            re.MULTILINE,
        )
        atoms = [a.strip().split()[:4] for a in atoms]
        for x, y, z, name in atoms:
            mol.add_atom(
                [float(x), float(y), float(z)], name, Chem.name_to_element(name), ""
            )

        iter = MolIterator(mol)

        return iter
