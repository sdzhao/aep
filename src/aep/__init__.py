"""AEP: a smooth map f_theta: R^d -> R^3 whose image is a 2D surface that preserves local geometry."""
from aep.api import fit_aep
from aep.model import AEPModel

__all__ = ["fit_aep", "AEPModel"]
__version__ = "0.1.0"
