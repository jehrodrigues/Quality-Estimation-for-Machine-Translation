# Quality Estimation for Machine Translation

Task goal: Assess the quality of a translation system/human without access to reference translations.

---

### Contents

* [Installation](#installation)
* [Data](#Data)
* [Train](#Train)
* [Evaluation](#Evaluation)
* [Visualization](#Visualization)
* [Experimentation](#Experimentation)

---

## Installation
```console
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Data

### Preprocessing text file

in order to get the necessary data to run the QE models

```console
$ python -m src.data.make_dataset
or
$ python -m src.data.make_dataset <language-pair>
```
Optionally, one can send the specific language pair to process as parameter (multiple options are possible). If no option is given, the script will process all the data available in the `data/raw` folder. These are the language pairs available in the data folder:

English-Arabic
```
en-ar
```
English-Italian
```
en-it
```
English-Japanese
```
en-jp
```
English-Korean
```
en-ko
```
Korean-English
```
ko-en
```
Korean-Japanese
```
ko-jp
```
Korean-Chinese
```
ko-zh
```
English-Russian
```
en-ru
```
Russian-English
```
ru-en
```
Russian-French
```
ru-fr
```
Russian-Chinese
```
ru-zh
```
English-German
```
en-de
```
English-French
```
en-fr
```
English-Chinese
```
en-zh
```
Spanish-Chinese
```
es-zh
```
Japanese-English
```
jp-en
```
Japanese-Korean
```
jp-ko
```
Japanese-Chinese
```
jp-zh
```
Portuguese-Chinese
```
pt-zh
```
Chinese-Italian
```
zh-it
```

## Train
in order to train QE models

```console
$ python -m src.models.train_model
or
$ python -m src.models.train_model <language-pair>
```
The free parameter can be set depending on language pair chosen (from the option above).

## Evaluation

in order to evaluate QE models

```console
$ python -m src.models.evaluate_model <language-pair>
```
The free parameter can be set depending on language pair chosen (from the option above).

## Visualization

in order to visualize QE models

```console
$ python -m src.visualization.visualize
or
$ python -m src.visualization.visualize <language-pair>

With Tensorboard
$ tensorboard --logdir 'runs/'
```
The free parameter can be set depending on language pair chosen (from the option above).

## Experimentation

All experimentation is kept in the notebooks.

```console
$ cd notebooks/
$ jupyter notebook
```


## Web Demo

A Dash application can provide a basic interface to upload files and run our quality evaluation methods.

```console
$ python src/webapp/app.py
```