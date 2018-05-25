import argparse
import os
from util.util import *


class TestOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument("--K", type=int, dest="K", default=10, help="Number of steps to observe from the past")
        self.parser.add_argument("--T", type=int, dest="T", default=10, help="Number of steps into the middle")
        self.parser.add_argument("--F", type=int, dest="F", default=10, help="Number of steps to observe from the future")
        self.parser.add_argument('--result_dir', type=str, default='./results', help='temporary results are saved here')
        self.parser.add_argument("--image_size", type=int,  nargs='+',  dest="image_size", default=[128], help="image size h w")
        self.parser.add_argument('--dataroot', required=True,  help='path to videos (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--textroot', required=True, help='path to trainings (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--video_list', type=str, default='test_data_list.txt', help='the name of the videolist file')
        self.parser.add_argument('--ckpt', type=str, default='network-lf.pytorch', help='the name of the ckpt to load')

        # Data Augment
        self.parser.add_argument('--data', required=True, type=str, help="name of test dataset [KTH|UCF]")
        self.parser.add_argument("--pick_mode", default='Slide', type=str, help="pick up clip [Random|First|Slide]")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if len(self.opt.image_size) == 1:
            a = self.opt.image_size[0]
            self.opt.image_size.append(a)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt.serial_batches = True
        self.opt.video_list = 'test_data_list.txt'
        self.opt.quant_dir = os.path.join(self.opt.result_dir, 'quantitative', self.opt.data, self.opt.name + '_' + str(self.opt.K) + '_' + str(self.opt.T))
        makedir(self.opt.quant_dir)
        self.opt.img_dir = os.path.join(self.opt.result_dir, 'images', self.opt.data,
                                          self.opt.name + '_' + str(self.opt.K) + '_' + str(self.opt.T))
        makedir(self.opt.img_dir)

        file_name = os.path.join(self.opt.quant_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt

