# pylint: disable=R0914
"""Module with plot related code"""

from typing import List, Optional, Any, cast

# np.set_printoptions(linewidth=120, precision=4)
import numpy as np
from bokeh.models import Band, ColumnDataSource, GlyphRenderer, tools
from bokeh.palettes import Category10_4 as palette  # pylint: disable=no-name-in-module
from bokeh.plotting import figure, show

from bokeh.plotting.figure import Figure


class GpChart:
    """
    A Chart using bokeh to represent a trained Gaussian Process.

    Use either the `show` method to show the chart, or call the `fig` method to
    returned a bokeh figure that you can tweaq and show yourselt.

    Note: Once created you can also set the x/y labels by setting the
    xlabel/ylabel properties.

    Parameters
    ----------
    source : ColumnDataSource, dict
        Either a bokeh ColumnDataSource of a dictionary containing at least the
        fields 'x', 'mu' and 'cov', where 'x' is a range of values, 'mu' is the
        predicted average function response for the range of values in 'x', and
        `cov` is covariance matrix.
    train_data_source : ColumnDataSource, None
        Optional ColumnDataSource containing a 'x' and 'y' columns with
        training data
    num_samples : int
        Number of samples (curves) of the multivariate Gaussian distribution
        with `mu` and `cov` that should be added to the plot. A maximum of four
        samples is possible. Default is zero
    plot_marker : bool
        If predicted points should be marked with a circle or not.
    xlabel : str
        Label of the 'x' axis. Default is "x"
    ylabel : str
        Label of the 'y' axis. Default is "y"
    kargs : dict
        The remaining arguments captured in `kargs` are passed to bokeh figure.
        Ex: width, height, title, etc.
    """
    def __init__(self,
                 source: ColumnDataSource,
                 train_data_source: Optional[ColumnDataSource] = None,
                 num_samples: int = 0,
                 plot_marker: bool = False,
                 xlabel: str = 'x',
                 ylabel: str = 'y',
                 **kargs: Any):
        # Number of samples currently added to the plot
        self._num_samples: int = 0
        self._sample_renderers: List[GlyphRenderer] = []

        if isinstance(source, dict):
            source = ColumnDataSource(data=source)

        x = source.data["x"]
        mu = source.data["mu"]
        cov_diag = np.diag(source.data["cov"])
        uncertainty = 1.96 * np.sqrt(cov_diag)

        assert (len(mu.shape) == 1)

        source.data["lower"] = mu - uncertainty
        source.data["upper"] = mu + uncertainty

        source.data["var"] = cov_diag

        # Get the appropriated minumum and maximum for the plot y range
        min_y, max_y = np.min(mu - uncertainty), np.max(mu + uncertainty)
        min_x, max_x = 1.02 * np.min(x), 1.02 * np.max(x)
        min_y, max_y = 1.05 * min_y, 1.05 * max_y

        # Note that we will add HoverTool later
        if "width" not in kargs:
            kargs["width"] = 900
        if "height" not in kargs:
            kargs["height"] = 400
        if "y_range" in kargs:
            fig = figure(x_range=(min_x, max_x), **kargs)
        else:
            fig = figure(x_range=(min_x, max_x),
                         y_range=(min_y, max_y),
                         **kargs)
        fig.xaxis.axis_label = xlabel
        fig.yaxis.axis_label = ylabel

        prediction_plot = fig.line("x",
                                   "mu",
                                   source=source,
                                   line_width=2,
                                   legend_label="Prediction Mean")
        if plot_marker:
            prediction_plot = fig.circle("x",
                                         "mu",
                                         source=source,
                                         legend_label="Prediction Mean")

        hover1 = tools.HoverTool()
        hover1.renderers = [prediction_plot]
        hover1.tooltips = [("x", "@x"), ("Mean", "@mu"),
                           ("Mean + 2σ", "@upper"), ("Mean - 2σ", "@lower"),
                           ("Var", "@var")]
        fig.add_tools(hover1)

        band = Band(base="x",
                    lower="lower",
                    upper="upper",
                    source=source,
                    level='image',
                    fill_color="#e0ffff",
                    fill_alpha=0.8)

        # Plot the training data
        if train_data_source is not None:
            cross_plot = fig.cross("x",
                                   "y",
                                   source=train_data_source,
                                   size=10,
                                   color="red",
                                   legend_label="Train")
            hover2 = tools.HoverTool()
            hover2.tooltips = [("Training Point", "(@x, @y)")]
            hover2.renderers = [cross_plot]
            fig.add_tools(hover2)

        fig.add_layout(band)
        fig.legend.click_policy = "hide"

        self._source: ColumnDataSource = source
        self._fig: Figure = fig

        if num_samples > 0:
            self.add_samples(num_samples)

    def show(self, notebook_handle: bool = False) -> Optional[Any]:
        """
        Show the chart

        Parameters
        ----------
        notebook_handle : bool
            Passed to bokeh `show` function. Set to True to return a notebook
            handler and allow updating the plot using a ipywidgets, for
            instance.
        """
        return show(self._fig, notebook_handle=notebook_handle)

    @property
    def fig(self) -> Figure:
        """Figure property"""
        return self._fig

    @property
    def xlabel(self) -> str:
        """Property to read/set the `x` axis label"""
        return cast(str, self._fig.xaxis.axis_label)

    @xlabel.setter
    def xlabel(self, new_value: str) -> None:
        self._fig.xaxis.axis_label = new_value

    @property
    def ylabel(self) -> str:
        """Property to read/set the `y` axis label"""
        return cast(str, self._fig.yaxis.axis_label)

    @ylabel.setter
    def ylabel(self, new_value: str) -> None:
        self._fig.yaxis.axis_label = new_value

    def update(self, mu: np.ndarray, cov: np.ndarray) -> None:
        """
        Update the plot

        Parameters
        ----------
        mu : np.ndarray
            The new mean values
        cov : np.ndarray
            The new covariance matrix
        """
        self._source.data['mu'] = mu
        self._source.data['cov'] = cov
        cov_diag = np.diag(self._source.data["cov"])
        uncertainty = 1.96 * np.sqrt(cov_diag)
        self._source.data["lower"] = mu - uncertainty
        self._source.data["upper"] = mu + uncertainty
        self._source.data["var"] = cov_diag

    def add_samples(self, num_samples: int) -> None:
        """
        Sample the GP and add a sample to the plot

        Parameters
        ----------
        num_samples : int
            Number of samples to add
        """
        self._num_samples = num_samples

        mu = self._source.data["mu"]
        cov = self._source.data["cov"]
        x = self._source.data["x"]

        samples = np.random.multivariate_normal(mu, cov, num_samples)
        # min_y, max_y = min(min_y, np.min(samples)), max(max_y, np.max(samples))

        for i, sample in enumerate(samples):
            self._sample_renderers.append(
                self._fig.line(x, sample, color=palette[i],
                               line_dash="dashed"))

    def remove_samples(self) -> None:
        """
        Remove the samples from the plot
        """
        for renderer in self._sample_renderers:
            self._fig.renderers.remove(renderer)
        self._num_samples = 0
        self._sample_renderers.clear()


# NOTE: Because the training data used to plot + signs and the "mu" data used
# to plot the prediction have different sizes, we need to create separated
# column data sources. For the same reason, we have to create two hover tools,
# one for the prediction data and one for the training data.
def plot_gp(source: ColumnDataSource,
            train_data_source: Optional[ColumnDataSource] = None,
            samples: Optional[np.ndarray] = None,
            plot_marker: bool = False,
            xlabel: str = 'x',
            ylabel: str = 'y',
            dont_show: bool = False,
            **kargs: Any) -> Optional[Figure]:
    """
    Plot a Gaussian Process.

    Parameters
    ----------
    source : ColumnDataSource, dict
        Either a bokeh ColumnDataSource of a dictionary containing at least the
        fields 'x', 'mu' and 'cov', where 'x' is a range of values,
        'mu' is the predicted average function response for the range of values
        in 'x', and `cov` is covariance matrix.
    train_data_source : ColumnDataSource, None
        Optional ColumnDataSource containing a 'x' and 'y' columns with
        training data
    samples : np.ndarray
        A 2D numpy array or an integer. In the first two cases each element is
        ploted by the 'x' values in `source`. If an int is passed, then a
        multivariate Gaussian distribution is generated using `mu` and `cov` in
        `source`. A maximum of four samples is possible.
    plot_marker : bool
        If predicted points should be marked with a circle or not.
    xlabel : str
        Label of the 'x' axis. Default is "x"
    ylabel : str
        Label of the 'y' axis. Default is "y"
    dont_show : bool
        If the plot should not be show and the figure should be returned instead. Default is False.
    kargs : dict
        The remaining arguments captured in `kargs` are passed to bokeh figure.
        Ex: width, height, title, etc.

    Returns
    -------
    figure or None
        Returns the bokeh figure if `dont_show` is True.
    """
    if isinstance(source, dict):
        source = ColumnDataSource(data=source)

    x: np.ndarray = source.data["x"]
    mu: np.ndarray = source.data["mu"]
    cov_diag: np.ndarray = np.diag(source.data["cov"])
    uncertainty: np.ndarray = 1.96 * np.sqrt(cov_diag)

    assert (len(mu.shape) == 1)

    source.data["lower"] = mu - uncertainty
    source.data["upper"] = mu + uncertainty

    source.data["var"] = cov_diag

    # Get the appropriated minumum and maximum for the plot y range
    min_y, max_y = np.min(mu - uncertainty), np.max(mu + uncertainty)
    min_x, max_x = 1.02 * np.min(x), 1.02 * np.max(x)

    if samples is not None:
        if isinstance(samples, int):
            cov = source.data["cov"]
            samples = np.random.multivariate_normal(mu, cov, samples)
        min_y, max_y = min(min_y, np.min(samples)), max(max_y, np.max(samples))
    min_y, max_y = 1.05 * min_y, 1.05 * max_y

    # Note that we will add HoverTool later
    if "width" not in kargs:
        kargs["width"] = 900
    if "height" not in kargs:
        kargs["height"] = 400
    if "y_range" in kargs:
        p = figure(x_range=(min_x, max_x), **kargs)
    else:
        p = figure(x_range=(min_x, max_x), y_range=(min_y, max_y), **kargs)
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    prediction_plot = p.line("x",
                             "mu",
                             source=source,
                             line_width=2,
                             legend_label="Prediction Mean")
    if plot_marker:
        prediction_plot = p.circle("x",
                                   "mu",
                                   source=source,
                                   legend_label="Prediction Mean")

    hover1 = tools.HoverTool()
    hover1.renderers = [prediction_plot]
    hover1.tooltips = [("x", "@x"), ("Mean", "@mu"), ("Mean + 2σ", "@upper"),
                       ("Mean - 2σ", "@lower"), ("Var", "@var")]
    p.add_tools(hover1)

    band = Band(base="x",
                lower="lower",
                upper="upper",
                source=source,
                level='image',
                fill_color="#e0ffff",
                fill_alpha=0.8)
    if samples is not None:
        for i, sample in enumerate(samples):
            p.line(x,
                   sample,
                   color=palette[i],
                   legend_label=f"Sample {i}",
                   line_dash="dashed")

    # Plot the training data
    if train_data_source is not None:
        cross_plot = p.cross("x",
                             "y",
                             source=train_data_source,
                             size=10,
                             color="red",
                             legend_label="Train")
        hover2 = tools.HoverTool()
        hover2.tooltips = [("Training Point", "(@x, @y)")]
        hover2.renderers = [cross_plot]
        p.add_tools(hover2)

    p.add_layout(band)
    p.legend.click_policy = "hide"

    if dont_show:
        return p

    show(p)
    return None
