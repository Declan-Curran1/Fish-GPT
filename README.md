############################################################## #OVERVIEW ##############################################################

Real-Time-Bushfire-Prediction
Takes high resolution multispectral sentinel-2 satellite imagery. Applies k-means clustering to group pixels into seven groups based on the 12 bands available with Sentinel 2. Uses time-series (data collected every 3 days between both Sentinel 2 satellites) on the Sydney/blue mountains area to predict whether clusters were predictors for fire. 'Fire' is defined as band 12 of the sentinel-2 satellites which measures Short Wave InfraRed (SWIR) light. This is an imperfect proxy for fires but works in this limited context. It is recommended to change this to another source such as NASA's freely available MODIS satellite data.

############################################################## #DATA ##############################################################

Datasource easily altered by choosing lat/lon co-ordinates + timeframe in 1st file
In the base files provided, data is trained & run on data in the greater sydney/blue mountains area with images collected on avg every 3 days. As this data is from the multispectral sentinel-2 satellites, a wide array of data is collected in each image (13 bands spanning RGB, short/near infrared, vegetation index etc.)

############################################################## #INSTRUCTIONS ##############################################################

Files should be run in the following order:
Sentinel-2 API-(sentinelhub)-Flow-Cleaned-1.py (Uses sentinelhub API to collect imagery)
K-Means-Unsupervised-Flow-Cleaned-2.py (Runs k-means clustering on all 13 bands across both satellites for Aus 2019 bushfires)
rasterio_polygonise-Flow-Cleaned-3.py ('polygonises' the classified images, translating the clusters into rasters in .shp file format)
Land-Use-Analyser-Flow-Cleaned-4.py (Uses land use analyser package to determine area of each cluster in 555m x 555m grid)
Final-Analyse-Flow-Cleaned-5.py (Using data across all grids + timeframes, is there a relationship between clusters & fires?)




![image](https://github.com/user-attachments/assets/a9660028-8e82-410f-96ea-47d4f7d3f55e)
