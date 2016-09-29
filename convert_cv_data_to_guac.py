import os
import json
import codecs
from optparse import OptionParser

import numpy as np
import pandas as pd


def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='prefix', default='css',
                      help='Prefix for naming items: default=%default')    
    parser.add_option('-n', dest='n_dev_folds', default=5,
                      help='Number of dev folds: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    prefix = options.prefix
    n_dev_folds = int(options.n_dev_folds)

    np.random.seed(42)

    input_filename = 'cv.json'

    with codecs.open(input_filename, 'r') as input_file:
    	items = json.load(input_file)

    index = 0
    names = []
    labels = []
    splits = []    
    sentences = {}
    for d in items:
    	name = prefix + '_' + str(index)
    	text = d['text']
    	label = int(d['y'])
    	split = int(d['split'])
    	names.append(name)
    	labels.append(label)
    	splits.append(split)
    	sentences[name] = text
    	index += 1

    minor = np.random.randint(0, n_dev_folds, len(splits))
    calibration = np.random.randint(0, 2, len(splits))

    base_dir = os.path.join('.', 'css', 'data')


    output_dir = os.path.join(base_dir, 'raw', 'text')
    if not os.path.exists(output_dir):
    	os.makedirs(output_dir)
    text_filename = os.path.join(output_dir, 'sentences.json')
    with codecs.open(text_filename, 'w', encoding='utf-8') as output_file:
    	json.dump(sentences, output_file, indent=2, encoding='utf-8')

    output_dir = os.path.join(base_dir, 'raw', 'labels')
    if not os.path.exists(output_dir):
    	os.makedirs(output_dir)
    labels_filename = os.path.join(output_dir, 'labels.csv')
    labels_df = pd.DataFrame(labels, index=names, columns=['Positive'])
    labels_df.to_csv(labels_filename)

    output_dir = os.path.join(base_dir, 'subsets')
    if not os.path.exists(output_dir):
    	os.makedirs(output_dir)    
    splits_filename = os.path.join(output_dir, 'splits.csv')
    splits_df = pd.DataFrame(np.vstack([splits, minor, calibration]).T, index=names, columns=['major_split', 'minor_split', 'calibration'])
    splits_df.to_csv(splits_filename)



if __name__ == '__main__':
    main()
