if [ ! -d ".data" ]; then
    mkdir .data
fi

python train.py AG_NEWS --device cpu --save-model-path  model.i --dictionary vocab.i --use-sp-tokenizer True --num-epochs 5 
# To run spm with YelpReviewFull
# python train.py YelpReviewFull --device cuda --save-model-path  model.i --dictionary vocab.i --use-sp-tokenizer True
cut -f 2- -d "," .data/AG_NEWS/test.csv | python predict.py  model.i  vocab.i --use-sp-tokenizer True > predict_script.o
