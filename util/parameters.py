import argparse
import os

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of celeba"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('epoch', type=int, default=100, help='the number of epoch')

    return parser.parse_args()