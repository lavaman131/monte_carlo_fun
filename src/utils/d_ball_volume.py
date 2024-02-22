import numpy as np
from scipy import stats, special


def area_square(d: int) -> float:
    """Find the area of a square in d dimensions.

    Parameters
    ----------
    d : int
        number of dimensions

    Returns
    -------
    float
        area of square in d dimensions
    """
    return 2**d


def volume_ball_monte_carlo(d: int, r: float, n: int, random_state: int = 42) -> float:
    """Find the volume of a ball in d dimensions.

    Parameters
    ----------
    d : int
        number of dimensions
    r : float
        radius of the ball
    n : int
        number of points to sample
    random_state : int, optional
        random seed, by default 42

    Returns
    -------
    float
        volume of ball in d dimensions
    """

    uniform = stats.uniform(-1, 2)

    x = uniform.rvs(size=(n, d), random_state=random_state)
    inside = np.sum(np.linalg.norm(x, axis=1) < r)
    return r**d * area_square(d) * inside / n


def volume_ball_analytical(d: int, r: float) -> float:
    """Find the volume of a ball in d dimensions.

    Parameters
    ----------
    d : int
        number of dimensions
    r : float
        radius of the ball

    Returns
    -------
    float
        volume of ball in d dimensions
    """
    return np.pi ** (d / 2) * r**d / special.gamma(d / 2 + 1)
