import numpy as np
from copy import deepcopy


class SingleFracModel:
    """
    This class calculate monitor well strain in a global coordinate system.
    It allows for multiple monitor wells, but a single fracture.
    """

    def __init__(self):
        self.dynamic_fracture = None
