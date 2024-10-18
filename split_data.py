# -*- coding: utf-8 -*-
"""
 @Author: Huidong Ma
 @E-mail: mahd@nbjl.nankai.edu.cn
 @DateTime: 2024/6/9 9:50
 @Description: args template.
"""
import sys
import time
import logging
import argparse
import numpy as np
import os

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-n', type=int)
    args = parser.parse_args(argv)
    return args

def main_ctrl(args):
    with open(args.input, 'rb') as f:  # 一次一个byte = 8bit
        series = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    args.prefix = os.path.basename(args.input).split('.')[0]

    vals = list(set(series))
    vals.sort()
    char2id_dict = {str(c): i for (i, c) in enumerate(vals)}
    id2char_dict = {str(i): c for (i, c) in enumerate(vals)}
    params = dict()
    segment_length = len(series) // args.n
    for i in range(args.n):
        start_index = i * segment_length
        # 最后一段可能包括剩余的部分
        end_index = start_index + segment_length if i < args.n - 1 else len(series)
        segment = series[start_index:end_index]
        fout = open(args.prefix + '.' + str(i), 'wb')
        fout.write(bytearray(segment))
        fout.close()
        params[args.prefix + '.' + str(i)] = len(segment)
    params['char2id_dict'] = char2id_dict
    params['id2char_dict'] = id2char_dict
    with open(args.prefix + '.params', 'w') as f:
        f.write(str(params))
    f.close()

def setupLogging(debug=False):
    logLevel = logging.DEBUG if debug else logging.INFO
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stderr, level=logLevel, format=logFormat)
    logging.info("Running %s" % " ".join(sys.argv))

def run(argv):
    setupLogging()
    args = parseArgs(argv)
    starttime = time.time()
    main_ctrl(args)
    logging.info("Finished in %0.2f seconds." % (time.time() - starttime))

if __name__ == '__main__':
    run(sys.argv[1:])
