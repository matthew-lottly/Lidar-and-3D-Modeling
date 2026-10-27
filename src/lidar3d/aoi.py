"""Study area: a small square in local UTM, chosen from the data (see docs/site-selection.md)."""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer

CENTER_LL = (-121.6615, 46.7065)        # chosen by scanning lidar canopy height, see docs/site-selection.md
HALF_M = 700.0                           # 1.4 km square
CRS = "EPSG:32610"                       # UTM 10N
RAINIER_SUMMIT = (-121.7603, 46.8529)    # for the distance statement only


@dataclass(frozen=True)
class AOI:
    cx: float
    cy: float
    half: float = HALF_M
    crs: str = CRS

    @classmethod
    def default(cls) -> "AOI":
        cx, cy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(*CENTER_LL)
        return cls(round(cx), round(cy))

    @property
    def bounds(self):                    # xmin, ymin, xmax, ymax in UTM
        return self.cx - self.half, self.cy - self.half, self.cx + self.half, self.cy + self.half

    @property
    def bbox_ll(self):                   # W, S, E, N with a small margin
        x0, y0, x1, y1 = self.bounds
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        pts = [t.transform(x, y) for x in (x0 - 30, x1 + 30) for y in (y0 - 30, y1 + 30)]
        return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)

    @property
    def size_m(self) -> float:
        return 2 * self.half
