from abc import abstractmethod

import problexity.classification as px

class ComplexityMeasure:
    @classmethod
    def compute(cls, X, y):
        pass

class F1v(ComplexityMeasure):
    @classmethod
    def compute(cls, X, y):
        return px.f1v(X, y)

class C2(ComplexityMeasure):
    @classmethod
    def compute(cls, X, y):
        return px.c2(X, y)