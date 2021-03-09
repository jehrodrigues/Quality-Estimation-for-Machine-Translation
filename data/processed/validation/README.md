DC Platform Data (Translation Validation)
========================================

Last update: August 31th, 2020

#Content

This folder is composed of parallel and tagged data from the Quality Validation stage, extracted from the DC Platform.

A folder is composed by:
```
project.txt
source.txt
target.txt
tags.txt
dataset.csv
train.csv
test.csv
```

where the files refer to the Quality Validation spreadsheet extracted from database:

Format:

```
project.txt
    <job_id> <hit_id> <project_title> <language_requirement>: JobId + HitId + Project Title + Main Language Requirement
source.txt
    <source_sentence>: Sentence in the source language
target.txt
    <target_sentence>: Translated Sentence in the target language
tags.txt
    <accuracy>:
    accuracy: 2. Is the meaning of the source conveyed? (true or false)
dataset.csv
    All data
train.csv
    Balanced data for training
test.csv
    Balanced data for validation
```