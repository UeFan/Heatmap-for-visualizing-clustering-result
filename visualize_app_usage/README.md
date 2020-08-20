# Example of visualizing phone app usage


Introduction
---------------
I collect the data of the mobile app usage in past two weeks of myself though the Screen Time in IOS. By using the new_heatmap, I can visualize both the time of using the apps and the time of first pickup the apps in one plot. I also separate the dates into weekdays and weekends so that I can better visualize the difference of my app usage in different time.

Result
---------------
The color of the cell indicates the time of using the app in one day. The size of the cell indicates the time of first pickup the app in one day, i.e. how many times in one day the user open the app immediately after unlocking the phone.
In the original data, the unit for the time is minute. For a better visualization, the data used in the following heatmap is `log( original data + 1 )`.
![Image text](https://github.com/UeFan/Heatmap-for-visualizing-clustering-result/blob/master/visualize_app_usage/p2.png)

Details
---------------
The original data is stored in phone_app_usage.xlsx. There are four parts of code in the .ipynb:
1. Read in the original data.
2. Process the data and compute the drendrogram.
3. Call new_heatmap.py to visualize the heatmap.
4. Compute the logarithm of the original data and visualize the result again.