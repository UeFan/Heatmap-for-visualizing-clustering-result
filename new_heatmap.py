from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib import axes
import matplotlib.transforms as mtrans
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def heatmap(data, row_labels, col_labels, ax_=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax_
        A `matplotlib.ax_es.Axes` instance to which the heatmap is plotted.  If
        not provided, use current ax_es or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax_:
        ax_ = plt.gca()

    # Plot the heatmap
    im = ax_.imshow(data, **kwargs)

    # # Create colorbar
    # cbar = ax_.figure.colorbar(im, ax_=ax_, aspect=40, shrink=0.5, **cbar_kw)
    # cbar.ax_.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax_.set_xticks(np.arange(data.shape[1]))
    ax_.set_xticklabels(col_labels)

    # Let the horizontal ax_es labeling appear on top.
    ax_.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_.get_xticklabels(), rotation=90, ha="right")

    # , rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax_.spines.items():
        spine.set_visible(False)

    ax_.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    #     ax_.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax_.grid(False, 'major')
    # ax_.yaxis.grid(True, 'minor')
    ax_.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax_.tick_params(which="minor", bottom=False, left=False)

    for sp in ax_.spines.values():
        sp.set_visible(True)

    return im

def annotate_heatmap(im, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "black"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """


    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    # else:
    #     threshold = 0 #im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center", fontsize=8)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if threshold is None:
                kw.update(color=textcolors[0])
                text = im.axes.text(j, i, data[i, j], **kw)
            else:
                kw.update(color=textcolors[int(im.norm(abs(data[i, j])) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)

            texts.append(text)

    return texts

def multi_group_heatmap(**kwargs):
    def matrix2scatter(x_, y_, c_, s_):
        x_list, y_list, c_list, s_list = [], [], [], []
        for j in range(y_.shape[0]):
            for i in range(x_.shape[0]):
                x_list.append(x_[i])
                y_list.append(y_[j])
                c_list.append(c_[j,i])
                s_list.append(s_[j,i])
        return x_list, y_list, c_list, s_list

    def clean_axis(ax):
        """Remove ticks, tick labels, and frame from axis"""
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    group0 = []
    if 'group0' in kwargs:
        group0 = kwargs['group0']
    group1 = []
    if 'group1' in kwargs:
        group1 = kwargs['group1']

    # ratio = len(x0) / (len(x0) + len(x1))
    # plot_grid = plt.GridSpec(1, 24, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    plot_grid = plt.GridSpec(5,7,wspace=0.0,hspace=0.0,width_ratios=[4,4,0.5,group0.shape[1],group1.shape[1],0.5,0.8], height_ratios=[1,3.5,0.5,3.5, 0.5])



    col_pairwise_dists_0 = []
    col_pairwise_dists_1 = []
    if 'col_pairwise_dists_0' in kwargs:
        col_pairwise_dists_0 = kwargs['col_pairwise_dists_0']
    if 'col_pairwise_dists_1' in kwargs:
        col_pairwise_dists_1 = kwargs['col_pairwise_dists_1']


    if_clean_axis = False
    if_color_bar = False
    if_size_bar = False

    if 'size_bar' in kwargs:
        if_size_bar = kwargs['size_bar']
    if 'clean_axis' in kwargs:
        if_clean_axis = kwargs['clean_axis']
    if 'color_bar' in kwargs:
        if_color_bar = kwargs['color_bar']

    if 'palette' in kwargs:
        palette = sns.color_palette(kwargs['palette'][0], kwargs['palette'][1])
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(np.min(group0),min(group1)), max(np.max(group0), max(group1)) # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(min(size),min(kwargs['size1'])), max(max(size),max(kwargs['size1']))
    marker = kwargs.get('marker', 's')
    size_scale = kwargs.get('size_scale', 500)
    kwargs_pass_on = {k: v for k, v in kwargs.items() if k not in [
        'group0', 'group1', 'palette', 'color_range', 'size0', 'size1', 'size_range', 'size_scale', 'marker', 'col_pairwise_dists_0', 'col_pairwise_dists_1',
        'group0_x_ticks', 'group1_x_ticks', 'y_order', 'xlabel', 'y_ticks','x_axis_label', 'chart_x_ticks', 'index_group_x_ticks', 'high_ligh_y_ticks',
        'ylabel', 'clean_axis', 'color_bar', 'size_bar', 'col_clusters0', 'col_clusters1', 'x_mean', 'y_mean', 'index_group', 'x_order_mean','chart'
    ]}
    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if (if_color_bar == True):
        if color_min < color_max:
            ax_cb = plt.subplot(plot_grid[1, 6])  # Use the rightmost column of the plot

            col_x = [0] * len(palette)  # Fixed x coordinate for the bars
            bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax_cb.barh(
                y=bar_y,
                width=[3] * len(palette),  # Make bars 5 units wide
                left=col_x,  # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0
            )
            ax_cb.grid(False)  # Hide grid
            ax_cb.set_facecolor('white')  # Make background white
            ax_cb.set_xticks([])  # Remove horizontal ticks
            ax_cb.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
            ax_cb.yaxis.tick_right()  # Show vertical ticks on the right

            ax_cb.spines['top'].set_visible(False)
            ax_cb.spines['right'].set_visible(False)
            ax_cb.spines['bottom'].set_visible(False)
            ax_cb.spines['left'].set_visible(False)

    if(if_size_bar == True):
        ax_sb = plt.subplot(plot_grid[3, 6])  # Use the rightmost column of the plot
        ax_sb.scatter(
            x=[0 for ii in range(11)],
            y=[jj for jj in range(11)],
            marker=marker,
            # s=[(0 + 0.3*kk)*size_scale/3 for kk in range(11)],
            s=[(ss*0.85/10 + 0.15)*size_scale for ss in range(11)],
            **kwargs_pass_on
        )
        ax_sb.set_xticks([])  # Remove horizontal ticks
        ax_sb.set_yticks(np.linspace(0, 10, 11))  # Show vertical ticks for min, middle and max
        # ax_sb.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
        ax_sb.set_yticklabels([int(size_min), ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', int(size_max)])

        ax_sb.yaxis.tick_right()  # Show vertical ticks on the right

        ax_sb.spines['top'].set_visible(False)
        ax_sb.spines['right'].set_visible(False)
        ax_sb.spines['bottom'].set_visible(False)
        ax_sb.spines['left'].set_visible(False)

    if(col_pairwise_dists_0 != [] and col_pairwise_dists_1 != []):
        # cluster
        col_clusters_0 = sch.linkage(col_pairwise_dists_0, method='complete')
        col_clusters_1 = sch.linkage(col_pairwise_dists_1, method='complete')

        ax_dg0 = plt.subplot(plot_grid[0,3])
        col_denD_0 = sch.dendrogram(col_clusters_0, color_threshold=np.inf, no_plot = False, ax = ax_dg0)
        clean_axis(ax_dg0)

        ax_dg1 = plt.subplot(plot_grid[0, 4])
        col_denD_1 = sch.dendrogram(col_clusters_1, color_threshold=np.inf, no_plot=False, ax=ax_dg1)
        clean_axis(ax_dg1)

##########################################################################

    x_mean_ticks = np.array(kwargs['index_group_x_ticks'])
    y_ticks = []
    index_group = []
    # if 'x_mean' in kwargs:
    #     x_mean = kwargs['x_mean']
    if 'y_ticks' in kwargs:
        y_ticks = kwargs['y_ticks']
    if 'index_group' in kwargs:
        index_group = kwargs['index_group']


    size = np.ones(index_group.shape[:])
    x, y, content, size = matrix2scatter(x_mean_ticks, y_ticks, index_group, size)


    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.95 / (size_max - size_min) + 0.05 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale

    x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    y_names = [t for t in y_ticks]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    ax_cb = plt.subplot(plot_grid[1:,0]) # Use the left 14/15ths of the grid for the main plot


    trans = mtrans.Affine2D().translate(4, 0)
    for t in ax_cb.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    set_yticks_ = [-0.5]
    last_y_groups_ = 0
    tt = 0
    for t in kwargs.get('y_order', ''):
        # print (t, int(t.split('_')[-1]))
        if (int(t.split('_')[-1]) != last_y_groups_):
            last_y_groups_ = int(t.split('_')[-1])
            set_yticks_.append(tt - 0.5)
        tt += 1
    ax_cb.set_yticks(set_yticks_, minor=True)

    ax_cb.set_yticks(np.arange(y_ticks.shape[0]), minor=False)
    ax_cb.set_yticklabels(y_ticks, minor=False)
    ax_cb.grid(False, 'major')
    ax_cb.yaxis.grid(True, 'minor')

    cmap = cm.get_cmap(kwargs['palette'][0], kwargs['palette'][1])
    im = heatmap(index_group, y, x, ax_=ax_cb,
                       cbarlabel="", cmap=cmap, aspect='auto', vmin=color_min, vmax=color_max)
    texts = annotate_heatmap(im, threshold=0.2)



    ax_cb.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax_cb.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax_cb.set_facecolor('white')
    if 'high_ligh_y_ticks' in kwargs:
        high_ligh_y_ticks = kwargs['high_ligh_y_ticks']
        for iiii in range(high_ligh_y_ticks[0]+1,high_ligh_y_ticks[1]+1):
            ax_cb.get_yticklabels()[-iiii].set_color("red")
##########################################################################

    if 'chart' in kwargs:
        chart = kwargs['chart']

        ax1 = plt.subplot(plot_grid[1:, 1])  # Use the left 14/15ths of the grid for the main plot
    
        trans = mtrans.Affine2D().translate(4, 0)
        for t in ax1.get_xticklabels():
            t.set_transform(t.get_transform() + trans)
    
        cmap_w = ListedColormap(['white'])
        chart_x_ticks = kwargs['chart_x_ticks']
        im = heatmap(np.zeros(chart.shape), y_names, chart_x_ticks, ax_=ax1,
                     cbarlabel="", cmap=cmap_w, aspect='auto')
        texts = annotate_heatmap(im, data=chart,)
    
        ax1.set_yticks(set_yticks_, minor=True)
        ax1.get_yaxis().set_ticks([])
        ax1.grid(False, 'major')
        ax1.yaxis.grid(True, 'minor')

        ax1.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax1.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
        ax1.set_facecolor('white')

##########################################################################


    if 'group0_x_ticks' in kwargs:
        x_names = np.array([t for t in kwargs['group0_x_ticks']])
    else:
        x_names = np.arange(group0.shape[1])
    if 'size0' in kwargs:
        size0 = kwargs['size0']
    else:
        size0 = [1] * len(x)
        
    x, y, content, size = matrix2scatter(x_names, y_ticks, group0, size0)
    



    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    ax2 = plt.subplot(plot_grid[1:, 3])  # Use the left 14/15ths of the grid for the main plot

    ax2.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in content],
        **kwargs_pass_on
    )

    ax2.set_xticks([v for k, v in x_to_num.items()])
    ax2.set_xticklabels([k for k in x_to_num], rotation=90, horizontalalignment='right')
    trans = mtrans.Affine2D().translate(4, 0)
    for t in ax2.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    ax2.set_xticks([t + 0.5 for t in ax_cb.get_xticks()], minor=True)
    ax2.set_yticks(set_yticks_, minor=True)

    ax2.grid(False, 'major')
    ax2.yaxis.grid(True, 'minor')

    ax2.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax2.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax2.set_facecolor('white')

    ax2.set_xlabel(kwargs.get('xlabel', ''))

    ax2.get_yaxis().set_ticks([])
    ax2.set_xlabel('\n  Group 0     ')
##########################################################################


    if 'group1_x_ticks' in kwargs:
        x_names = np.array([t for t in kwargs['group1_x_ticks']])
    else:
        x_names = np.arange(group1.shape[1])
    if 'size1' in kwargs:
        size1 = kwargs['size1']
    else:
        size1 = [1] * len(x)
    x, y, content, size = matrix2scatter(x_names, y_ticks, group1, size1)



    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    ax2 = plt.subplot(plot_grid[1:, 4])  # Use the left 14/15ths of the grid for the main plot
    marker = kwargs.get('marker', 's')



    ax2.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in content],
        **kwargs_pass_on
    )

    ax2.set_xticks([v for k, v in x_to_num.items()])
    ax2.set_xticklabels([k for k in x_to_num], rotation=90, horizontalalignment='right')
    trans = mtrans.Affine2D().translate(6, 0)
    for t in ax2.get_xticklabels():
        t.set_transform(t.get_transform() + trans)


    ax2.set_xticks([t + 0.5 for t in ax_cb.get_xticks()], minor=True)
    ax2.set_yticks(set_yticks_, minor=True)



    ax2.grid(False, 'major')
    ax2.yaxis.grid(True, 'minor')


    ax2.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax2.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax2.set_facecolor('white')

    ax2.set_xlabel(kwargs.get('xlabel', ''))

    ax2.get_yaxis().set_ticks([])
    x_axis_label = ''
    if 'x_axis_label' in kwargs:
        x_axis_label = kwargs['x_axis_label']
    ax2.set_xlabel(x_axis_label  +'\n Group 1     ')



def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

