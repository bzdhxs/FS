"""HHO (Harris Hawks Optimization) feature selection.

Default parameters are defined as class attributes. To permanently change them,
edit the values below. For temporary overrides, use config.yaml's algo_params.
"""

from mealpy.swarm_based.HHO import OriginalHHO
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector


@register_algorithm("HHO")
class HHOSelector(BaseMealpySelector):
    default_epoch = 200
    default_pop_size = 250
    default_penalty = 0.2

    def create_optimizer(self):
        return OriginalHHO(epoch=self.epoch, pop_size=self.pop_size)
