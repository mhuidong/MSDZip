import shutil
import sys
import time
import logging
import argparse
import torch
import os

import compress_model
import arithmeticcoding_fast
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Source file.')
    parser.add_argument('output', type=str, help='Compressed file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
    parser.add_argument('--tempdir', '-T', type=str, help='Temporary folder name.')
    parser.add_argument('--prefix', '-p', type=str, help='Prefixes of files')
    parser.add_argument('--index', '-i', type=str, help='Index of files')
    parser.add_argument('--batchsize', '-b', type=int, default=512, help='Sample size in one batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-7, help='Weight decay.')
    parser.add_argument('--timesteps', type=int, default=16, help='The number of history symbols')
    parser.add_argument('--vocab_dim', type=int, default=16, help='The dimension of vocab.')
    parser.add_argument('--vocab_size', type=int, default=256, help='The size of vocab.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='The dimension of hidden layer.')
    parser.add_argument('--ffn_dim', type=int, default=4096, help='The dimension of ffn layer.')
    parser.add_argument('--layers', type=int, default=5, help='The number of layers.')
    parser.add_argument('--seed', type=int, default=0, help='Random seeds.')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--load', action='store_true', help='Load the model')
    parser.add_argument('--ratio', type=float, default=0.05, help='Pretrain ratio.')
    args = parser.parse_args(argv)
    return args


def compress(args, temp_file, series, train_data, final):
    bs, ts = args.batchsize, args.timesteps
    f = [open(temp_file + '.' + str(i), 'wb') for i in range(bs)]  # 创建batchsize个空文件

    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]  # 同DZip
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

    prob = np.ones(args.vocab_size) / args.vocab_size  # 初始化一个概率       vocab_size = 256  len(prob)=256
    cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)  # 累加        266
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    iter_num = len(train_data) // bs  # 多少批
    ind = np.array(range(bs)) * iter_num  # 批index
    # train_data = reorder_data(train_data, bs, iter_num)
    iter_num -= ts
    for i in range(bs):
        for j in range(ts):
            enc[i].write(cumul, series[ind[i] + j])
    cumul_batch = np.zeros((bs, args.vocab_size + 1), dtype=np.uint64)  # [128, 256+1]  # 原来是vocab_size
    model = compress_model.MixedModel(batchsize=args.batchsize, layers=args.layers, hidden_dim=args.hidden_dim,
                                      ffn_dim=args.ffn_dim, vocab_size=args.vocab_size,
                                      vocab_dim=args.vocab_dim, timesteps=ts).cuda()  # 没有用到vocab_dim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    flag = 0
    if args.load:
        logging.info('Loading Model!')
        model.load_state_dict(torch.load(args.prefix + '_model/{}.{}.pth'.format(args.prefix, int(args.index)-1)))
        # model.load_state_dict(torch.load('modelpath/model.pth'))
    for train_index in range(iter_num):
        model.train()
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).cuda().long()  # 128 * 33
        logits = model.forward(train_batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], train_batch[:, -1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        for i in range(bs):
            enc[i].write(cumul_batch[i, :], y[i])
        ind += 1

        if train_index == int(iter_num * args.ratio) and args.save:
            logging.info('Saving Model!')
            torch.save(model.state_dict(), args.prefix + '_model/{}.{}.pth'.format(args.prefix, args.index))
            # torch.save(model.state_dict(), 'modelpath/model.pth')

        # if train_index >= (iter_num * 0.1 * flag):
        #     logging.info('{:^3.0f}%: {}'.format(10 * flag, loss.item() / np.log(2)))
        #     flag += 1
    logging.info('Compreesion finished.')

    for i in range(bs):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if final is not None:
        logging.info("last series")
        f = open(temp_file + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(args.vocab_size) / args.vocab_size
        cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(len(final)):
            enc.write(cumul, final[j])
        logging.info("Last encode part don't need inference.")

        enc.finish()
        bitout.close()
        f.close()
    return


def main(args):
    t1 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.prefix:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]

    if not args.tempdir:
        args.tempdir = "{}_bs{}_ts{}_v{}_h{}_f{}_l{}".format(args.prefix+'.'+args.index, args.batchsize, args.timesteps,
                                                             args.vocab_dim, args.hidden_dim, args.ffn_dim,
                                                             args.layers)
    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    temp_file = args.tempdir + '/compressed_temp_file'

    # args.timesteps = args.timesteps * (args.hidden_dim // args.vocab_dim)
    # Read input source file, and record key information
    with open(args.input, 'rb') as f:  # 一次一个byte = 8bit
        series = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()

    params = eval(open(args.prefix + '.params', 'r').read())
    char2id_dict = params['char2id_dict']
    series = np.array([char2id_dict[str(c)] for c in series])

    # Generating training data
    train_data = strided_app(series, args.timesteps + 1, 1)

    # Stat vocab freq

    total_num = len(train_data)  # sentence的个数
    if total_num % args.batchsize == 0:  # 正好够整数个bs
        compress(args, temp_file, series, train_data, None)
    else:  # 不够整数个batchsize
        ini_num = total_num // args.batchsize * args.batchsize  # 只压缩整数批的数据，整数个批里面有l+timesteps个元素
        # print(1, ini_num+args.timesteps)
        compress(args, temp_file, series[:ini_num + args.timesteps], train_data[:ini_num], series[ini_num:])

    # Combined compressed results
    f = open(args.output, 'wb')
    for i in range(args.batchsize):
        f_in = open(temp_file + '.' + str(i), 'rb')
        byte_str = f_in.read()  # 写入的二进制文件，固定写入，记住就行了
        byte_str_len = len(byte_str)  # 长度
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    if total_num % args.batchsize != 0:
        f_in = open(temp_file + '.last', 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f.close()

    total = 0
    for ff in os.listdir(args.tempdir):
        total += os.path.getsize(args.tempdir + '/' + ff)

    # Remove temp file
    shutil.rmtree(args.tempdir)
    t2 = time.time()
    f1_size, f2_size = os.stat(args.input).st_size, os.stat(args.output).st_size
    # logging.info('Compressed File Size: {} Bytes'.format(round(f2_size, 5)))
    # logging.info('Compression Ratio: {}'.format(round(f2_size / f1_size * 8, 5)))
    logging.info('Compression Time: {} secs'.format(round(t2 - t1, 5)))
    # logging.info('Peak GPU memory usage: {} KBs'.format(torch.cuda.max_memory_allocated() // 1024))
    logging.info(
        'The params are:\nbatchsize\tlr\thidden_dim\tvocab_dim\tffn_dim\tlayers\ttimesteps\tvocab_size\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            args.batchsize, args.lr, args.hidden_dim, args.vocab_dim, args.ffn_dim, args.layers,
            args.timesteps, args.vocab_size))


def setupLogging(debug=False):
    logLevel = logging.DEBUG if debug else logging.INFO
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stderr, level=logLevel, format=logFormat)
    logging.info("Running %s" % " ".join(sys.argv))


def run(argv):
    setupLogging()
    args = parseArgs(argv)
    starttime = time.time()
    main(args)
    logging.info("Finished in %0.2f seconds." % (time.time() - starttime))


if __name__ == '__main__':
    run(sys.argv[1:])
