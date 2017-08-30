This is a folder of several extractive models: fasttext, cnn_extractor, sequence-to-sequence based extractive model.

* Train and test fasttext model as a classifier

```
python run/fasttext_train.py run/fasttext_train.xml
python run/fasttext_text.py run/fasttext_test.xml
```

* Use fasttext to extract sentences

```
python run/run_fasttext_extractor.py run/run_fasttext_extractor.xml
```

* Train and test cnn_extractor as a classifier

```
python run/naive_extractor_train.py run/naive_extractor_train.xml
python run/naive_extractor_test.py run/naive_extractor_test.xml
```

* Use cnn_extractor to extract sentences

```
python run/run_naive_extractor.py run/run_naive_extractor.xml
```

* Train and test sequence-to-sequence based extractive model as a classifier

```
python run/sentence_extract_train.py run/sentence_extract_train.xml
python run/sentence_extract_test.py run/sentence_extract_test.xml
```

* Use sequence-to-sequence based extractive model to extract sentences

```
python run/run_sentence_extractor.py run/run_sentence_extractor.xml
```

* Launch a model (can be either type of network above), listen to the port 8100 of localhost, waiting for the input query (URL) to that address and return the results.

```
python main.py run/laucher.xml
```

