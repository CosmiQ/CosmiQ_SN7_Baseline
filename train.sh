cp -r $1 train
mkdir -p tmp/model tmp/raw tmp/grouped
python src/preproc.py
python src/csv.py train
python src/train.py
