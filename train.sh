cp -r $1 train
python src/preproc.py
python src/train.py
