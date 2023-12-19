rm -rf tests
mkdir tests
cd tests
wget https://files.rcsb.org/view/1XDN.pdb
cat 1XDN.pdb | grep -v ATP > receptor.pdb
cat 1XDN.pdb | grep ATP > ligand.pdb

# Remove terminal phosphate
cat ligand.pdb | grep -v "O1G" | grep -v "PG" | grep -v "O2G" | grep -v "O3G" > ligand2.pdb

cd ../
python deepfrag.py --receptor tests/receptor.pdb --ligand tests/ligand2.pdb --cx 44.807 --cy 16.562 --cz 14.092


