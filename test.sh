cp -r $1 test

python src/infer.py
python src/postproc.py

mv inference_out/sn7_1_preds/csvs/sn7_baseline_predictions.csv $2
