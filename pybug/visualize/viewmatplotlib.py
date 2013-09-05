import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import abc
from pybug.visualize.base import Renderer


class MatplotlibRenderer(Renderer):
    r"""
    Abstract class for rendering visualizations using Matplotlib.

    Parameters
    ----------
    figure_id : int or ``None``
        A figure id or ``None``. ``None`` assumes we maintain the Matplotlib
        state machine and use ``plt.gcf()``.
    new_figure : bool
        If ``True``, creates a new figure to render on.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        super(MatplotlibRenderer, self).__init__(figure_id, new_figure)

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of ``self.figure_id`` and
        ``self.new_figure``. If ``self.figure_id == None`` then ``plt.gcf()``
        is used. ``self.figure_id`` is also set to the correct id of the figure
        if a new figure is created.

        Returns
        -------
        figure : Matplotlib figure object
            The figure we will be rendering on.
        """
        if self.new_figure or self.figure_id is not None:
            self.figure = plt.figure(self.figure_id)
        else:
            self.figure = plt.gcf()

        self.figure_id = self.figure.number

        return self.figure


class MatplotlibImageViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, image):
        super(MatplotlibImageViewer2d, self).__init__(figure_id, new_figure)
        self.image = image

    def _render(self, **kwargs):
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            im = self.image
            im = im if len(im.shape) == 2 else im[..., 0]
            plt.imshow(im, cmap=cm.Greys_r, **kwargs)
        else:
            plt.imshow(self.image, **kwargs)

        return self


class MatplotlibPointCloudViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, points):
        super(MatplotlibPointCloudViewer2d, self).__init__(figure_id,
                                                           new_figure)
        self.points = points

    def _render(self, image_view=False, cmap=None,
                      colour_array='b', label=None, **kwargs):
        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        plt.scatter(points[:, 0], points[:, 1], cmap=cmap,
                    c=colour_array, label=label)
        return self


class MatplotlibTriMeshViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, points, trilist):
        super(MatplotlibTriMeshViewer2d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render(self, image_view=False, label=None, **kwargs):
        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        plt.triplot(points[:, 0], points[:, 1], self.trilist,
                    label=label, color='b')

        return self


class MatplotlibLandmarkViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, label, landmark_dict):
        super(MatplotlibLandmarkViewer2d, self).__init__(figure_id, new_figure)
        self.label = label
        self.landmark_dict = landmark_dict

    def _plot_landmarks(self, include_labels, image_view, **kwargs):
        colours = kwargs.get('colours',
                             np.random.random([3, len(self.landmark_dict)]))
        halign = kwargs.get('halign', 'center')
        valign = kwargs.get('valign', 'bottom')
        size = kwargs.get('size', 10)

        # TODO: Should we enforce viewing landmarks with Matplotlib? How
        # do we do this?
        # Set the default colormap, assuming that the pointclouds are
        # also viewed using Matplotlib
        kwargs.setdefault('cmap', cm.jet)

        for i, (label, pc) in enumerate(self.landmark_dict.iteritems()):
            # Set kwargs assuming that the pointclouds are viewed using
            # Matplotlib
            kwargs['colour_array'] = [colours[:, i]] * pc.n_points
            kwargs['label'] = '{0}_{1}'.format(self.label, label)
            pc.view_on(self.figure_id, image_view=image_view, **kwargs)

            if include_labels:
                ax = plt.gca()
                points = pc.points[:, ::-1] if image_view else pc.points
                for i, p in enumerate(points):
                    ax.annotate(str(i), xy=(p[0], p[1]),
                                horizontalalignment=halign,
                                verticalalignment=valign, size=size)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    def _render(self, include_labels=True, **kwargs):
        self._plot_landmarks(include_labels, False, **kwargs)
        return self


class MatplotlibLandmarkViewer2dImage(MatplotlibLandmarkViewer2d):

    def __init__(self, figure_id, new_figure, label, landmark_dict):
        super(MatplotlibLandmarkViewer2dImage, self).__init__(
            figure_id, new_figure, label, landmark_dict)

    def _render(self, include_labels=True, **kwargs):
        self._plot_landmarks(include_labels, True, **kwargs)
        return self