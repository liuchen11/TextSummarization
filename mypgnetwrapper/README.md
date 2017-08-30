This is a folder of several abstractive models: sequence-to-sequence based abstractive model, pointer-generator model, pointer-generator network with sent2vec based loss.

* Train and test sequence-to-sequence network or pointer-generator network

```
python run/run_train_wrapper.py --data_path=<data_path> --vocab_path=<vocab_path> --mode=[train/decode] --log_root=log --exp_name=<experiment name> --pointer_gen=[True/False] --coverage=[True/False]
```

* Train and test pointer-generator network with sent2vec-based loss. The template external file is available on `run/external.xml`

```
python run/run_train_wrapper.py --data_path=<data_path> --vocab_path=<vocab_path> --mode=[train/decode] --log_root=log --exp_name=<experiment name> --external_config=<external config file>
```

* Train a mixed model, keep the extractive part fixed and train the abstractive part based on it. Here parameter sentence_extract_root is the root folder of sentence extractor code. In addition, sentence_extract_config is the xml file that config the extractive model, like `laucher.xml`

```
python run/run_mixed_train_wrapper.py --data_path=<data_path> --vocab_path=<vocab_path> --mode=train --log_root=log --exp_name=<experiment name> --sentence_extract_root=../sentenceextract --sentence_extract_config=<config file for extractive model>
```

* The decoding phrase of a mixed model

```
python run/run_mixed_decode_wrapper.py --data_path=<data_path> --vocab_path=<vocab_path> --mode=decode --log_root=log --exp_name=<experiment name> --sententce_extract_root=../sentenceextract --sentence_extract_config=<config file for extractive model> --article_folder=<folder to save the article> --refer_folder=<folder to save the gold summary> --output_folder=<folder to save the decoded summary>
```

There are two tools to evaluate the results of abstractive model. One is to calculate the rouge scores, the other is used to calculate the similarity of embeddings from different documents

```
python util/calc_rouge.py <output_file> <ground_truth folder> <output folder> [<ground_truth suffix>] [output suffix]
python util/calc_vec.py util/calc_vec.xml
```
