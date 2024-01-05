from enum import Enum


class ModelType(Enum):
    CLASSIFIER = 1
    DENSE_PREDICTOR = 2
    SSL = 3