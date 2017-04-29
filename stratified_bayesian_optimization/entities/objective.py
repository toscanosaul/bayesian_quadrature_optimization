from __future__ import absoluste_import

from schematics.models import Model
from schematics.types import FloatType
from schematics.types.compound import ListType


class Objective(Model):
    evaluated_points = ListType(FloatType)
    objective_values = ListType(FloatType)
    # This is for noisy evaluations
    standard_deviation_evaluations = ListType(FloatType)
