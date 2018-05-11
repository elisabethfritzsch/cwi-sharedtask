These are files for performing complex word identification. 
This repository contains
- cwi.py: then main script file. Run this file to perform the complex word identification. To be able to run, the datasets for the task containing Spanish and English data should be placed in a folder named "datasets", in the same directory as cwi.py
- utils: a folder with supplementary scripts and data

utils contains:
- model.py: a script for creating and training a classifier
- scorer.py: a script for scoring a trained classifier
- dataset.py: a script for parsing the datasets
- greek_and_latin_affixes.txt: a list of greek and latin affixes
- affixes.py: a script to parse this list of affixes
- english_freqs.pkl: a pickled dictionary mapping English words to their absolute frequencies in a large corpus
- english_tf.pkl: a pickled dictionary mapping English trigrams to their absolute frequencies in a large corpus
- spanish_freqs.pkl: a pickled dictionary mapping Spanis words to their absolute frequencies in a large corpus
- spanish_tf.pkl: a pickled dictionary mapping Spanish trigrams to their absolute frequencies in a large corpus
- frequencies.py: a script for creating the 4 pickled dictionaries described above (included for completeness)

