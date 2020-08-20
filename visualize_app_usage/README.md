# Example of visualizing phone app usage


Introduction
---------------
The color of the cell indicates the time of using the app in one day. The size of the cell indicates the time of first pickup the app in one day, i.e. how many times in one day the user open the app immediately after unlocking the phone.
In the original data, the unit for the time is minute. For a better visualization, the data used in the following heatmap is log( original data + 1 ).
![Image text](https://github.com/UeFan/Heatmap-for-visualizing-clustering-result/blob/master/visualize_app_usage/p2.png)


The original data is stored in phone_app_usage.xlsx. There are four parts of code in the .ipynb:
1. Read in the original data.
2. Process the data and compute the drendrogram.
3. Call new_heatmap.py to visualize the heatmap.
4. Compute the logarithm of the original data and visualize the result again.