# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:58:20 2022

@author: Yuwei Wang
"""

import tarfile

with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()