# A (Heavily Documented) TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

## **Major History**
  * Started adversarial training
  * Found errors in Attention Mechanism, lack of scheduled sampling, and `eval.py` when using multiple gpus.  Still not fixed.
  * Around 16 hours of training on gtx 1060 giving male mel-spec as input and having linear-scale and mel-scale as outputs yeilds this training curve.

<img src="fig/mean_loss.png">

The mel-scale and linear scale loss look like

<img src="fig/mean_loss1.png">

<img src="fig/mean_loss2.png">

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.2
  * librosa
  * tqdm

## Data
See root download script.

[The World English Bible](https://en.wikipedia.org/wiki/World_English_Bible) is a public domain update of the American Standard Version of 1901 into modern English. Its text and audio recordings are freely available [here](http://www.audiotreasure.com/webindex.htm). Unfortunately, however, each of the audio files matches a chapter, not a verse, so is too long for many machine learning tasks. I had someone slice them by verse manually. You can download [the audio data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and its [text](https://dl.dropboxusercontent.com/u/42868014/text.csv) from my dropbox.



## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepare_pavoque.py` creates sliced sound files from raw sound data, and constructs necessary information.
  * `prepro.py` loads vocabulary, training/evaluation data.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `utils.py` has several custom operational functions.
  * `modules.py` contains building blocks for encoding/decoding networks.
  * `networks.py` has three core networks, that is, encoding, decoding, and postprocessing network.
  * `train.py` is for training on paired speech.
  * `train_gan.py` is for training advarsarially using LSGAN
  * `eval.py` is for sample synthesis.
  * `eval_gan.py` is ofr sample synthesis using gan_model
  

## Training
  * STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 2. Download CMU Artic data.
  * STEP 3. Run `train.py`. or `train_multi_gpus.py` if you have more than one gpu.

## Sample Synthesis
  * Run `eval.py` to get samples.

### Acknowledgements
I would like to show my respect to Dave, the host of www.audiotreasure.com and the reader of the audio files.
