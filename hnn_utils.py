'''
HNN utilities
'''
def plot_mean_lfp(self, *, trial_no=None, contact_no=None, tmin=None, tmax=None,
             ax=None, decim=None, color='cividis', voltage_offset=50,
             voltage_scalebar=200, show=True, mean=False):
    """mean_across_trials : bool
        If True, average the voltage traces across all selected trials before plotting."""
    
    from hnn_core.viz import plot_laminar_lfp
    from hnn_core.extracellular import _get_laminar_z_coords
    
    if trial_no is None:
        plot_data = self.voltages
    elif isinstance(trial_no, (list, tuple, int, slice)):
        plot_data = self.voltages[trial_no, ]
    else:
        raise ValueError(f'unknown trial number type, got {trial_no}')

    if isinstance(contact_no, (list, tuple, int, slice)):
        plot_data = plot_data[:, contact_no, ]
    elif contact_no is not None:
        raise ValueError(f'unknown contact number type, got {contact_no}')

    contact_labels, _ = _get_laminar_z_coords(self.positions)

    # If mean is True, average the trials
    if mean==True:
        plot_data = plot_data.mean(axis=0)  # Calculate the mean of the trials along the first axis
        fig = plot_laminar_lfp(
                self.times, plot_data, tmin=tmin, tmax=tmax, ax=ax,
                decim=decim, color=color,
                voltage_offset=voltage_offset,
                voltage_scalebar=voltage_scalebar,
                contact_labels=contact_labels,
                show=show)
    else:
        for trial_data in plot_data:
            fig = plot_laminar_lfp(
                self.times, trial_data, tmin=tmin, tmax=tmax, ax=ax,
                decim=decim, color=color,
                voltage_offset=voltage_offset,
                voltage_scalebar=voltage_scalebar,
                contact_labels=contact_labels,
                show=show)
    
    return fig

def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    NB copied from :func:`mne.viz.utils.plt_show`.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)

def plot_mean_csd(self, vmin=None, vmax=None, interpolation='spline',
             sink='b', colorbar=True, ax=None, show=True, mean=False):
    """Plot laminar current source density (CSD) estimation
    mean : bool
    If True, average the voltage traces across all selected trials before plotting.
    """
    from hnn_core.viz import plot_laminar_csd
    from hnn_core.extracellular import _get_laminar_z_coords, calculate_csd2d

    if mean==True:
        lfp_ = self.voltages
        lfp = lfp_.mean(axis=0)
    else: 
        lfp = self.voltages[0]
        
    contact_labels, delta = _get_laminar_z_coords(self.positions)
    contact_labels=[int(x) for x in contact_labels]
    
    csd_data = calculate_csd2d(lfp_data=lfp,
                               delta=delta)

    fig = plot_laminar_csd(self.times, csd_data,
                               contact_labels=contact_labels, ax=ax,
                               colorbar=colorbar, vmin=vmin, vmax=vmax,
                               interpolation=interpolation, sink=sink,
                               show=show)

    return fig