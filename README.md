# 2-Selmer-statistics
Code to establish positive proportion results for 2-Selmer ranks of elliptic curves over Q.

To run this code, you need Python and the 'nauty' graph software installed, specifically 'geng' and 'labelg.'

Install dependencies:
- Nauty: 'brew install nauty' (Mac) or 'sudo apt-get install nauty' (Linux)
- Python: pip install numpy scipy networkx

Run with:
python hypercube_solver.py

The script outputs results to results_cubes.txt and results_relations.txt. Edit the 'dimension_end' variable in the file to change how many dimensions it processes.
