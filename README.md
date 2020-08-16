# Heatmap for visualizing clustering result
![Image text](https://github.com/UeFan/Heatmap-for-visualizing-clustering-result/blob/master/example.png)



Introduction
---------------

new_heatmap.py is a wrapped python function used for generating heatmaps that are suitable for visualizing clustering result. 
The heatmap generated has the following unique features:
It can show an extra data dimension by setting different cell sizes in the heatmap. Additionally, 

1. The heatmap is able to show two clusters. They are seperated horizontally in the heatmap. They share the same y-axis, which is usually the data points used in clusetering.
2. Except that the cell color of the heatmap can be set individually, the proposed heatmap can alter the individual cell size. This allows us to show an extra dimension of the data in the heatmap.
3. The orders of the row and column can be ranked by the hierarchical order given by scipy.cluster.hierarchy. The resulting dendrograms (illustrate the hierarchical order) are also shown in the heatmap.


Parameters
---------------

### **`multi_group_heatmap(**kwargs)`**

**Parameters**:

**`y_ticks`** : A 1-d np.array containing the yticks in order.

**`index_group`** : A 2-d np.array containing the values for the color of heatmap located on the left (which is the "heatmap of index group"). The number of rows should be same as the length of `y_ticks`. The number of columns should be same as the length of `index_group_x_ticks`.

**`index_group_x_ticks`** : A 1-d np.array containing the ordered xticks of the index group heatmap.

**`group0`** : A 2-d np.array containing the values for the color of heatmap located on the middle right (which is the "heatmap of group0"). The number of rows should be same as the length of `y_ticks`. The number of columns should be same as the length of `group0_x_ticks`.

**`size0`** : A 2-d np.array containing the values for the cell size of heatmap located on the middle right (which is the "heatmap of group0"). The number of rows should be same as the length of `y_ticks`. The number of columns should be same as the length of `group0_x_ticks`.

**`group0_x_ticks`** : A 1-d np.array containing the ordered xticks of the group0 heatmap.

**`group1`** : A 2-d np.array containing the values for the color of heatmap located on the middle right (which is the "heatmap of group1"). The number of rows should be same as the length of `y_ticks`. The number of columns should be same as the length of `group1_x_ticks`.

**`size1`** : A 2-d np.array containing the values for the cell size of heatmap located on the middle right (which is the "heatmap of group1"). The number of rows should be same as the length of `y_ticks`. The number of columns should be same as the length of `group1_x_ticks`.

**`group1_x_ticks,`** : A 1-d np.array containing the ordered xticks of the group1 heatmap.

**`color_range`** : A tuple `(size_min, size_max)` that enables capping the values of `size` being applied to the shapes in the plot. Essentially controls min and max size of the shapes. 

**`size_range`** : Used to scale the size of the shapes in the plot to make them fit the size of the fields in the matrix. Default value is 500. You will likely need to fiddle with this parameter in order to find the right value for your figure size and the size range applied.

**`palette`** : A list of colors to use as the heatmap palette. The values from `color` are mapped onto the palette so that `min(color) -> palette[0]` and `max(color) -> palette[len(palette)-1]`, and the values in between are linearly interpolated. A good way to choose or create a palette is to simply use Seaborn palettes (https://seaborn.pydata.org/tutorial/color_palettes.html).

**`chart`** : A 2-d np.array containing the values to be filled in the char located on the right side of the index group. 

**`chart_x_ticks`** : A 1-d np.array containing the ordered xticks of the chart.

**`x_axis_label`** : Label for the x-axis.

**`high_ligh_y_ticks`** : A tuple `(starting row number, ending row number)` that sets the starting and ending row numbers of the yticks that will be colored in red.

**`col_pairwise_dists_0`** : It is the return of sch.linkage(...), indicating the pairwise distances of datapoints in group0. It helps to generate the hierarchical clustering results on y-axis and the dendrogram. If not defined, the dendrogram will not be displayed.

**`col_pairwise_dists_0`** : It is the return of sch.linkage(...), indicating the pairwise distances of datapoints in group1. It helps to generate the hierarchical clustering results on y-axis and the dendrogram. If not defined, the dendrogram will not be displayed.

**`color_bar`** : A bool controls whether show the color bar (legend of cell color) or not.

**`size_bar`** : A bool controls whether show the size bar (legend of cell size) or not.


Example
---------------

Please refer to the example.ipynb

Acknowledgment
---------------

I’d like to acknowledge the assistance of my supervisors Prof. Mathias Unberath and Dr. Bela Turk.


Reference
---------------

The heatmap.py is modified from **heatmapz** https://pypi.org/project/heatmapz/.
More details about heatmapz can be found at https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

