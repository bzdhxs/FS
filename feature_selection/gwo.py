"""GWO (Grey Wolf Optimizer) feature selection.

Default parameters are defined as class attributes. To permanently change them,
edit the values below. For temporary overrides, use config.yaml's algo_params.
"""

from mealpy.swarm_based.GWO import OriginalGWO
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector


@register_algorithm("GWO")
class GWOSelector(BaseMealpySelector):
    default_epoch = 200
    default_pop_size = 100
    default_penalty = 0.2

    def create_optimizer(self):
        return OriginalGWO(epoch=self.epoch, pop_size=self.pop_size)
