"""GA (Genetic Algorithm) feature selection.

Default parameters are defined as class attributes. To permanently change them,
edit the values below. For temporary overrides, use config.yaml's algo_params.
"""

from mealpy.evolutionary_based.GA import BaseGA
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector


@register_algorithm("GA")
class GASelector(BaseMealpySelector):
    default_epoch = 200
    default_pop_size = 50
    default_penalty = 0.6
    default_pc = 0.85
    default_pm = 0.05

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        super().__init__(target_col, band_range, logger, **kwargs)
        self.pc = kwargs.get('pc', self.default_pc)
        self.pm = kwargs.get('pm', self.default_pm)

    def create_optimizer(self):
        return BaseGA(epoch=self.epoch, pop_size=self.pop_size, pc=self.pc, pm=self.pm)
