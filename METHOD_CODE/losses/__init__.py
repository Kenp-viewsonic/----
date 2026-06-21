from .evo_loss import EvoLoss
from .stable_plastic_reg import StablePlasticRegLoss
from .open_loss import OpenSetLoss
from .orth_loss import OrthogonalityLoss
from .separation_loss import NewClassSeparationLoss
from .class_margin_loss import ClassMarginLoss

__all__ = ["EvoLoss", "StablePlasticRegLoss", "OpenSetLoss", "OrthogonalityLoss", "NewClassSeparationLoss", "ClassMarginLoss"]
