cp -r $1 test
python src/csv.py test
python src/infer.py
python src/postproc.py
mv tmp/solution.csv $2
