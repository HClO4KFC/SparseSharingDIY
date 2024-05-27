import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml', type=str, default='default.yaml')

    in_args, unk = parser.parse_known_args()
    if len(unk) != 0:
        print("warning: unknown args ", unk)
    return in_args


def str2list(s:str, conj:str)->list:
    lst = [item.strip() for item in s.split(conj) if item.strip()]
    return lst
