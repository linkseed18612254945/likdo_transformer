#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : container.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from utils import io
import numpy as np
import os


class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError("Not contain attr {}".format(k))
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def print(self, sep=': ', end='\n', file=None):
        return kvprint(self, sep=sep, end=end, file=file)

    def save_as_json(self, save_path):
        io.dump_json(save_path, self)


class GView(object):
    def __init__(self, dict_=None):
        if dict_ is None:
            dict_ = dict()
        object.__setattr__(self, '_dict', dict_)

    def __getattr__(self, k):
        if k not in self.raw():
            raise AttributeError
        return self.raw()[k]

    def __setattr__(self, k, v):
        self.raw()[k] = v

    def __delattr__(self, k):
        del self.raw()[k]

    def __getitem__(self, k):
        return self.raw()[k]

    def __setitem__(self, k, v):
        self.raw()[k] = v

    def __delitem__(self, k):
        del self.raw()[k]

    def __contains__(self, k):
        return k in self.raw()

    def __iter__(self):
        return iter(self.raw().items())

    def raw(self):
        return object.__getattribute__(self, '_dict')

    def update(self, other):
        self.raw().update(other)

    def copy(self):
        return GView(self.raw().copy())

    def print(self, sep=': ', end='\n', file=None):
        return kvprint(self.raw(), sep=sep, end=end, file=file)



def kvprint(data, indent=0, sep=' : ', end='\n', max_key_len=None, file=None):
    if len(data) == 0:
        return
    def format_printable_data(data):
        t = type(data)
        if t is np.ndarray:
            return 'ndarray{}, dtype={}'.format(data.shape, data.dtype)
        # Handle torch.tensor
        if 'Tensor' in str(t):
            return 'tensor{}, dtype={}'.format(tuple(data.shape), data.dtype)
        elif t is float:
            return "{:.6f}".format(data)
        else:
            return str(data)

    keys = sorted(data.keys())
    lens = list(map(len, keys))
    if max_key_len is not None:
        max_len = max_key_len
    else:
        max_len = max(lens)
    for k in keys:
        print('  ' * indent, end='')
        print(k + ' ' * max(max_len - len(k), 0), format_printable_data(data[k]), sep=sep, end=end, file=file, flush=True)


class BasicConfig(G):
    def __init__(self):
        super(BasicConfig, self).__init__()
        self.data = G()
        self.model = G()
        self.train = G()
        self.predict = G()

