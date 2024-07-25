if [ ! -d ".data" ]; then
    mkdir .data
fi

python train.py AG_NEWS --device cpu --save-model-path  model.i --dictionary vocab.i
cut -f 2- -d "," .data/AG_NEWS/test.csv | python predict.py  model.i  vocab.i > predict_script.o

# To train using pre-trained sentencepiece tokenizer
# python train.py AG_NEWS --device cpu --save-model-path  model.i --dictionary vocab.i --use-sp-tokenizer True

# To run spm with YelpReviewFull
# python train.py YelpReviewFull --device cuda --save-model-path  model.i --dictionary vocab.i --use-sp-tokenizer True
