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

heatmap(**kwargs)

to be continue……

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

