# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .GroundingDINO import build_groundingdino


def build_model(args):
    """Return the GroundingDINO model/criterion/postprocessors tuple."""
    return build_groundingdino(args)
