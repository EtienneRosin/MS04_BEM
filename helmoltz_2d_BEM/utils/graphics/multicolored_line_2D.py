"""
@file multicolored_line_2D.py
@brief Implementation of a MulticolorLine2d class that is used to create multicolored line
@note I mainly took the examples from https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html and https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html and adapted them
@author Etienne Rosin 
@version 0.1
@date 28/09/2024
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.colors import Normalize
import matplotlib as mpl

import cmasher as cmr


class MulticolorLine2d(LineCollection):
    """
    Represents a multicolored 2D line whose coloring is based on a third value, a colormap and a norm. 
    
    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.
    """
    def __init__(self, x, y, z, cmap, norm = None, **lc_kwargs):
        """
        Initialize the line.

        Parameters
        ----------
        x, y : array-like
            The horizontal and vertical coordinates of the data points.
        z : array-like
            The data to color the line, which should be the same size as x and y.
        cmap : str
            The colormap to use (works with cmasher colormaps)
        **lc_kwargs
            Any additional arguments to pass to matplotlib.collections.LineCollection
            constructor. This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.
        """
        if "array" in lc_kwargs:
            warnings.warn('The provided "array" keyword argument will be overridden')

        # Default the capstyle to butt so that the line segments smoothly line up
        default_kwargs = {"capstyle": "butt"}
        default_kwargs.update(lc_kwargs)

        segments = self.create_segments(x, y)
        z = np.asarray(z)
        if norm is None:
            norm = plt.Normalize(z.min(), z.max())
        super().__init__(segments = segments, cmap = cmap, norm = norm, **lc_kwargs)
        self.set_array(z)
        
        
    def create_segments(self, x, y):
        """
        Create the line segments

        Parameters
        ----------
        x, y : array-like
            The horizontal and vertical coordinates of the data points.
            
        Returns
        -------
        array-like
            The generated segments.
        """
        # Compute the midpoints of the line segments. Include the first and last points
        x = np.asarray(x)
        y = np.asarray(y)
        x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
        y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

        # Determine the start, middle, and end coordinate pair of each line segment.
        # Use the reshape to add an extra dimension so each pair of points is in its
        # own list. Then concatenate them to create:
        # [
        #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
        #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
        #   ...
        # ]
        coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
        return segments


class HandlerMulticolorLine2d(HandlerLineCollection):
    """
    Create a custom legend handler for a MulticolorLine2d object.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def create_artists(self, legend: plt.legend, artist: plt.Artist, xdescent: float, ydescent: float, width: float, height: float, fontsize: float, trans: mpl.transforms.Transform):
        """
        Create the artists for the legend.

        Returns
        -------
        MulticolorLine2d
            The MulticolorLine2d of the legend
        """

        x = np.linspace(0, width, self.get_numpoints(legend) + 1)
        y = np.zeros(self.get_numpoints(legend) + 1) + height / 2. - ydescent

        lc_kwargs = dict(linewidth = artist.get_linewidth(), alpha = artist.get_alpha(), linestyle = artist.get_linestyle())
        lc = MulticolorLine2d(x = x, y = y, z = x, cmap=artist.cmap, transform=trans, **lc_kwargs)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]



if __name__ == "__main__":
    cmap = "cmr.iceburn"
    x_data = np.logspace(1, 3, 50)
    y_data = np.random.rand(3, len(x_data))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x_data, y_data[0], c = "blue", label = "fzfiiuyb")
    color_line = MulticolorLine2d(x = x_data, y = y_data[0], z = range(len(x_data)), cmap = cmap, linewidth=4, alpha = 0.5, label = "hihi")
    ax.add_collection(color_line)


    legend = ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})

    plt.show()
