"""波数領域音響反射係数 C_r = 0 に基づく無反射境界条件（NRBC）の FEM 実装。

日本音響学会 2026 年秋季研究発表会 2-Q-1
星加 慧・岩見 貴弘・尾本 章（九州大学）
"""

from . import config, fem, boundary, solvers, metrics  # noqa: F401

__version__ = "1.0.0"
__all__ = ["config", "fem", "boundary", "solvers", "metrics"]
