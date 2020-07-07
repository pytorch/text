# This is an example to create a machine translation dataset and train a translation model.

Users will use the training data in the raw file from Multi30k dataset to train a machine translation model with the character composition method.

To try the example, simply run the following commands:

```bash
python train_char.py
```

For character level training, and

```bash
python train_word.py
```

For word level training

## Experiment Result

The following is the output example for running `train_char.py`

```
Epoch: 01 | Time: 2m 10s
	Train Loss: 5.277 | Train PPL: 195.798 |  Train BLEU:   0.001
	 Val. Loss: 4.088 |  Val. PPL:  59.598 |  Val. BLEU:   0.006
Epoch: 02 | Time: 2m 29s
	Train Loss: 3.711 | Train PPL:  40.877 |  Train BLEU:   0.022
	 Val. Loss: 2.964 |  Val. PPL:  19.369 |  Val. BLEU:   0.048
Epoch: 03 | Time: 2m 32s
	Train Loss: 2.901 | Train PPL:  18.189 |  Train BLEU:   0.055
	 Val. Loss: 2.172 |  Val. PPL:   8.774 |  Val. BLEU:   0.111
Epoch: 04 | Time: 2m 46s
	Train Loss: 2.391 | Train PPL:  10.927 |  Train BLEU:   0.092
	 Val. Loss: 1.766 |  Val. PPL:   5.849 |  Val. BLEU:   0.164
Epoch: 05 | Time: 2m 40s
	Train Loss: 2.085 | Train PPL:   8.042 |  Train BLEU:   0.118
	 Val. Loss: 1.503 |  Val. PPL:   4.494 |  Val. BLEU:   0.196
Epoch: 06 | Time: 2m 39s
	Train Loss: 1.856 | Train PPL:   6.398 |  Train BLEU:   0.140
	 Val. Loss: 1.302 |  Val. PPL:   3.678 |  Val. BLEU:   0.229
Epoch: 07 | Time: 2m 40s
	Train Loss: 1.683 | Train PPL:   5.383 |  Train BLEU:   0.157
	 Val. Loss: 1.164 |  Val. PPL:   3.202 |  Val. BLEU:   0.250
Epoch: 08 | Time: 2m 44s
	Train Loss: 1.554 | Train PPL:   4.730 |  Train BLEU:   0.168
	 Val. Loss: 1.075 |  Val. PPL:   2.930 |  Val. BLEU:   0.263
Epoch: 09 | Time: 2m 38s
	Train Loss: 1.455 | Train PPL:   4.283 |  Train BLEU:   0.178
	 Val. Loss: 1.016 |  Val. PPL:   2.763 |  Val. BLEU:   0.271
Epoch: 10 | Time: 2m 46s
	Train Loss: 1.373 | Train PPL:   3.948 |  Train BLEU:   0.187
	 Val. Loss: 0.972 |  Val. PPL:   2.644 |  Val. BLEU:   0.280
| Test Loss: 1.011 | Test PPL:   2.748 |  Test BLEU:   0.273
```

And the following is the output of `train_word.py`
# TODO