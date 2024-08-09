
from .aggregator import MaxminAggregator, MeanAggregator,MaxMinAggregation
from .bert_aggregator import BertAggregation

all = {
    'MaxminAggregator': MaxminAggregator,
    'MeanAggregator': MeanAggregator,
    'MaxMinAggregation': MaxMinAggregation,
    'BertAggregation': BertAggregation,
}