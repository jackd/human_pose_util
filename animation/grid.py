from __future__ import division
import numpy as np
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.graphics.collections import SegmentCollection


def grid(transform, viewport, scale=1.0):
    ticks = SegmentCollection(
        mode="agg", transform=transform, viewport=viewport, linewidth='local',
        color='local')
    labels = GlyphCollection(transform=transform, viewport=viewport)
    # xmin,xmax = 0,800
    # ymin,ymax = 0,800
    xmin, xmax = -1, 1
    nx = 11
    dx = (xmax - xmin) / (nx - 1)
    ymin, ymax = -1, 1
    ny = 11
    dy = (ymax - ymin) / (ny - 1)

    z = 0

    regular = FontManager.get("OpenSans-Regular.ttf")
    bold = FontManager.get("OpenSans-Bold.ttf")

    labels_scale = 0.001
    for i, x in enumerate(np.linspace(xmin, xmax, nx)):
        text = "%.2f" % (x*scale)
        labels.append(text, regular, origin=(x, ymin - dy / 2, z),
                      scale=labels_scale, direction=(1, 0, 0),
                      anchor_x="center", anchor_y="top")

    for i, y in enumerate(np.linspace(ymin, ymax, ny)):
        text = "%.2f" % (y*scale)
        labels.append(text, regular, origin=(xmin - dx / 2, y, z),
                      scale=labels_scale, direction=(1, 0, 0),
                      anchor_x="right", anchor_y="center")

    title = "H3M example"
    labels.append(title, bold, origin=(0, xmax + dx, z),
                  scale=2 * labels_scale, direction=(1, 0, 0),
                  anchor_x="center", anchor_y="center")

    # Frame
    # -------------------------------------
    P0 = [(xmin, ymin, z), (xmin, ymax, z), (xmax, ymax, z), (xmax, ymin, z)]
    P1 = [(xmin, ymax, z), (xmax, ymax, z), (xmax, ymin, z), (xmin, ymin, z)]
    ticks.append(P0, P1, linewidth=2)

    # Grids
    # -------------------------------------
    n = 11
    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))

    P0[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P0[:, 1] = ymin
    P0[:, 2] = z
    P1[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P1[:, 1] = ymax
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1, color=(0, 0, 0, .25))

    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))
    P0[:, 0] = xmin
    P0[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P0[:, 2] = z
    P1[:, 0] = xmax
    P1[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1, color=(0, 0, 0, .25))

    # Majors
    # -------------------------------------
    n = 11
    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))
    P0[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P0[:, 1] = ymin - 0.015
    P0[:, 2] = z
    P1[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P1[:, 1] = ymin + 0.025 * (ymax - ymin)
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1.5)
    P0[:, 1] = ymax + 0.015
    P1[:, 1] = ymax - 0.025 * (ymax - ymin)
    ticks.append(P0, P1, linewidth=1.5)

    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))
    P0[:, 0] = xmin - 0.015
    P0[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P0[:, 2] = z
    P1[:, 0] = xmin + 0.025 * (xmax - xmin)
    P1[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1.5)
    P0[:, 0] = xmax + 0.015
    P1[:, 0] = xmax - 0.025 * (xmax - xmin)
    ticks.append(P0, P1, linewidth=1.5)

    # Minors
    # -------------------------------------
    n = 111
    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))
    P0[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P0[:, 1] = ymin
    P0[:, 2] = z
    P1[:, 0] = np.linspace(xmin, xmax, n)[1:-1]
    P1[:, 1] = ymin + 0.0125 * (ymax - ymin)
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1)
    P0[:, 1] = ymax
    P1[:, 1] = ymax - 0.0125 * (ymax - ymin)
    ticks.append(P0, P1, linewidth=1)

    P0 = np.zeros((n - 2, 3))
    P1 = np.zeros((n - 2, 3))
    P0[:, 0] = xmin
    P0[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P0[:, 2] = z
    P1[:, 0] = xmin + 0.0125 * (xmax - xmin)
    P1[:, 1] = np.linspace(ymin, ymax, n)[1:-1]
    P1[:, 2] = z
    ticks.append(P0, P1, linewidth=1)
    P0[:, 0] = xmax
    P1[:, 0] = xmax - 0.0125 * (xmax - xmin)
    ticks.append(P0, P1, linewidth=1)
    return ticks
