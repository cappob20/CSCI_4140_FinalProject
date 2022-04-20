import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from FinalProject import train, test, interactive

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'interactive'), help='what to run')
    parser.add_argument('--work_dir', help='model location', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    #ensure work dir exists
    if not os.path.isdir(args.work_dir):
        print('Making working directory {}'.format(args.work_dir))
        os.makedirs(args.work_dir)

    model_path = os.path.join(args.work_dir, "saved_weights.h5")

    #"saved_weights.h5"
    if args.mode == 'train':
        train(args.work_dir)
    elif args.mode == 'test':
        test(model_path, args.test_data, args.test_output)
    elif args.mode == 'interactive':
        interactive(model_path)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
