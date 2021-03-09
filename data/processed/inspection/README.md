DC QA Spreadsheets (Translation Inspection)
========================================

Last update: August 31th, 2020

#Content

This folder is composed of parallel and tagged data from the Quality Inspection stage, extracted from the QA Spreadsheets.

A folder is composed by:
```
error_category.txt
error_type.txt
severity.txt
source.txt
target.txt
gold.txt
```

where the files refer to the Quality Inspection spreadsheet provided by the Translation Services Team:

Format:

```
error_category.txt
    <error_category>: Accuracy, Fluency, Terminology, Style, Design, Locale_Convention, Verity
error_type.txt
    <error_type>:
    Accuracy: Addition, Omission, Mistranslation, Over-translation, Under-translation, Untranslated-text, Improper-exact-TM-match
    Fluency: Punctuation, Spelling, Grammar, Grammatical register, Inconsistency, Link/cross-reference, Character encoding
    Terminology: Inconsistent with termbase, Inconsistent use of terminology
    Style: Awkward, Company style, Inconsistent style, Third-party style, Unidiomatic
    Design: Length, Local formatting, Markup, Missing text, Truncation/text expansion
    Locale_Convention: Address format, Date format, Currency format, Measurement format, Shortcut key, Telephone format
    Verity: Culture-specific reference
severity.txt
    <severity>: Major or Minor
source.txt
    <source_sentence>: Sentence in the source language
target.txt
    <target_sentence>: Translated Sentence in the target language
gold.txt
    <gold_sentence>: Post edited Sentence in the target language
```