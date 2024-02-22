from abc import abstractmethod
import numpy as np
from scipy import stats
from typing import Callable


class Prior:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def p(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rvs(self, size: int, random_state: int) -> np.ndarray:
        pass


class UniformPrior(Prior):
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def p(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, fill_value=1 / (self.b - self.a))

    def rvs(self, size: int, random_state: int) -> np.ndarray:
        return stats.uniform(self.a, self.b - self.a).rvs(
            size=size, random_state=random_state
        )


class NormalPrior(Prior):
    def __init__(self, loc: float, scale: float) -> None:
        self.loc = loc
        self.scale = scale

    def p(self, x: np.ndarray) -> np.ndarray:
        return stats.norm(loc=self.loc, scale=self.scale).pdf(x)  # type: ignore

    def rvs(self, size: int, random_state: int) -> np.ndarray:
        return stats.norm(loc=self.loc, scale=self.scale).rvs(
            size=size, random_state=random_state
        )


class MonteCarlo:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def set_target_distribution(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.func_to_integrate = func

    def initialize(
        self,
        lower_bound: float,
        upper_bound: float,
        prior: Prior,
        n_steps: int = 1000,
    ) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_steps = n_steps
        self.prior = prior

    def sample(self) -> float:
        if self.lower_bound == self.upper_bound:
            return 0.0
        x = self.prior.rvs(size=self.n_steps, random_state=self.random_state)
        mask = (x >= self.lower_bound) & (x <= self.upper_bound)
        y = self.func_to_integrate(x)
        # importance sampling
        return np.mean(mask * y / self.prior.p(x))  # type: ignore
