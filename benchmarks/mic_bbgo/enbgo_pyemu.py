# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:38:01 2024

Scripts for experimenting EnBGO

@author: mac732
"""

import pandas as pd
import numpy as np
import glob
import os
import datetime
import shutil
from distutils.dir_util import copy_tree
import pyemu

pyemu.os_utils.start_workers("template","pestpp-mou","mic_enbgo.pst",num_workers=4,worker_root=".",master_dir="master")      