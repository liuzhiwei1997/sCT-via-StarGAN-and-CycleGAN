"""Compatibility wrappers for the CycleGAN networks.

The original training code imports ``Generator`` and ``Discriminator`` from a
``model`` module, but only ``CycleGAN.py`` was committed to the repository.
This file restores that import path while preserving the constructor signature
used throughout ``solver_val.py``.
"""

from CycleGAN import Discriminator as _Discriminator
from CycleGAN import Generator as _Generator


class Generator(_Generator):
    """Adapter matching the historical StarGAN-like constructor signature."""

    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        del c_dim, repeat_num
        super().__init__(in_channels=1, features=conv_dim)


class Discriminator(_Discriminator):
    """Adapter matching the historical StarGAN-like constructor signature."""

    def __init__(self, image_size=512, conv_dim=64, c_dim=3, repeat_num=6):
        del image_size, c_dim, repeat_num
        super().__init__(in_channels=1, features=conv_dim)
