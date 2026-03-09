"""MPA (Marine Predators Algorithm) feature selection.

Default parameters are defined as class attributes. To permanently change them,
edit the values below. For temporary overrides, use config.yaml's algo_params.
"""

from mealpy.swarm_based.MPA import OriginalMPA
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector


@register_algorithm("MPA")
class MPASelector(BaseMealpySelector):
    default_epoch = 250
    default_pop_size = 120
    default_penalty = 0.4

    def create_optimizer(self):
        return OriginalMPA(epoch=self.epoch, pop_size=self.pop_size)
