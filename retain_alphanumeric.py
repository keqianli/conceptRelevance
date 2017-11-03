import os
import re
import logging
import sys

file = '../data/data_oneFilePerLine/jmlr_vldb/texts.txt'


if len(sys.argv) > 1:
    file = sys.argv[1]

file_out = file+'_alphanumeric'

if len(sys.argv) > 2:
    file_out = sys.argv[2]


def processSingleLine(string, outputQueue):
    # remove all except alphanumeric, concat, underscore
    string = re.sub(r"[^A-Za-z0-9_\-<>/]", " ", string)
    # merge consecutive spaces
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string


def processByLineSameOutput(inputFile, processSingleLine, outputFile=None):
    if not outputFile:
        outputFile = inputFile+'_processed'
    with open(os.path.join(outputFile), 'w') as f_out:
        with open(inputFile) as f:
            for l in f:
                try:
                    f_out.write(str(processSingleLine(l.strip(), None)).strip()+'\n')
                except Exception, e:
                    logging.debug(e)


if __name__ == '__main__':
    processByLineSameOutput(file, processSingleLine, file_out)
