source activate Python36
rm __target__/*
transcrypt --ecom make_grid.py
cp index.html __target__/
cd __target__/
~/simple_server
