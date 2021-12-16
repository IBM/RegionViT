# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

def add_config(cfg):
    """
    Add config for RegionViT.
    """
    # MODEL
    cfg.MODEL.BACKBONE.DP_RATE = 0.2
    cfg.MODEL.BACKBONE.DET_NORM = True
    cfg.MODEL.BACKBONE.REGIONVIT = 'regionvit_small_224'


    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAMW"


    # INPUT
    cfg.INPUT.AUG_MODE = "FIXED"



