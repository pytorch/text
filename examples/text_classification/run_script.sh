if [ ! -d ".data" ]; then
    mkdir .data
fi

python train.py AG_NEWS --device cuda --save-model-path  model.i --dictionary vocab.i
cut -f 2- -d "," .data/ag_news_csv/test.csv | python predict.py  model.i  vocab.i > predict_script.o
