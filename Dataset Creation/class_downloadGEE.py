# Importing libraries

from pathlib import Path
import datetime
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio


import requests

from rasterio.plot import show
import matplotlib.pyplot as plt


# Function to convert a merged shapefile to a single Earth Engine Geometry
def shapefile_to_ee_geometry(shapefile_path):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure the CRS is WGS84 (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Merge all polygons into a single geometry
    merged_geometry = gdf.union_all()

    # Convert the merged geometry to GeoJSON format
    merged_geojson = gpd.GeoSeries([merged_geometry]).to_json()
    
    # Extract the geometry part from the GeoJSON
    merged_geom_dict = gpd.GeoSeries([merged_geometry]).__geo_interface__["features"][0]["geometry"]

    # Convert the GeoJSON geometry to an Earth Engine Geometry
    ee_geometry = ee.Geometry(merged_geom_dict)
    
    return ee_geometry

class S2_download(object):
    
    def __init__(self, start_date, end_date, cloud_cover, bands, ee_geometry, output_dir):
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover
        self.bands = bands
        self.ee_geometry = ee_geometry
        self.output_dir = output_dir
    
    # Function to mask clouds using Cloud Score+ logic
    def mask_s2_clouds(self, image):
        """Masks clouds in a Sentinel-2 image using the QA band.

        Args:
            image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: A cloud-masked Sentinel-2 image.
        """
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask).divide(10000)
    
    def s2_processed(self, year):
        """
        Returns a pre-processed Sentinel-2 Level 1C image collection.
        The image collection is filtered by date, cloud cover, and masked for clouds.
        
        args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            cloud_cover (int): The maximum cloud cover percentage as a decimal.
            bands (list): The list of bands to select.
        returns:
            ee.ImageCollection: The pre-processed Sentinel-2 image collection.
        """
        dataset = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterDate(self.start_date, self.end_date)
        # Pre-filter to get less cloudy granules.
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_cover))
        .map(self.mask_s2_clouds)
        .select(self.bands)
        )
        return dataset

             
        
    # Function to create and export monthly composites
    def export_monthly_composites(self, year):
        
        s2 = self.s2_processed(year)
            
        for month in range(7, 9):
            start = datetime.date(year, month, 1)
            end = (datetime.date(year, month + 1, 1) if month < 12 else datetime.date(year + 1, 1, 1))
            monthly_composite = (s2.filterDate(str(start), str(end))
                                .median()
                                .clip(self.ee_geometry)
                                .set('month', month, 'year', year))
            
            # Export the composite
            task = ee.batch.Export.image.toDrive(
                image=monthly_composite,
                description=f'Sentinel2_Monthly_{year}_{month:02d}',
                folder=self.output_dir,
                scale=10,
                region=self.ee_geometry.getInfo()['coordinates'],
                crs='EPSG:4326',
                maxPixels=1e13  # Increase this if needed
            )
            
            task.start()
            print(f"Export started for {year}-{month:02d}!")




class gedi_download(object):
    
    def __init__(self, start_date, end_date, ee_geom, output_dir):
        self.start_date = start_date
        self.end_date = end_date
        self.ee_geom = ee_geom
        self.output_dir = output_dir
        
        
        # Function to select highest quality GEDI data
    def quality_mask(self, image):
        return image.updateMask(image.select('l4_quality_flag').eq(1)) \
                    .updateMask(image.select('degrade_flag').eq(0))

    # Function to mask unreliable GEDI measurements with relative standard error > 50%
    def error_mask(self, image):
        relative_se = image.select('agbd_se').divide(image.select('agbd'))
        return image.updateMask(relative_se.lte(0.5))

    # Function to mask GEDI measurements on slopes > 30%
    def slope_mask(self, image, slope):
        return image.updateMask(slope.lt(30))


    def process_gedi(self, ):
        # Define the GEDI Aboveground Biomass dataset
        gedi_dataset = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY") \
            #.select("agbd", "agbd_se")  # Select the aboveground biomass (AGB) band

        # Filter the GEDI data by date and region
        # Filter GEDI data by date and geometry
        gedi_filtered = gedi_dataset.filter(ee.Filter.date(self.start_date, self.end_date)) \
                            .filter(ee.Filter.bounds(self.ee_geom))

        # Get slope data from SRTM
        srtm = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(srtm)

        # Apply masks
        gedi_processed = gedi_filtered.map(self.quality_mask) \
                                    .map(self.error_mask) \
                                    .map(lambda img: self.slope_mask(img, slope))


        image_list = gedi_processed.toList(gedi_processed.size())
        
        return gedi_processed, image_list

    # Function to export each image in the collection
    def export_image(self, image, index):
        """Exports an image as a task to Google Drive."""
        image_name = image.getInfo()['properties']['system:index']
        #print(image_name)
    
            
        task = ee.batch.Export.image.toDrive(
            image=image.select('agbd'),
            description= image_name,
            folder=self.output_dir,
            fileNamePrefix=image_name,
            region=self.ee_geom.coordinates().getInfo(),
            scale=30,  # Resolution in meters
            crs="EPSG:4326",  # CRS
            maxPixels=1e15  # Increase this if needed
        )
        task.start()
        #print(f"Export task for {image_name} started.")
    
    def export_all_images(self):
        try:
            gedi_processed, image_list = self.process_gedi()
            
            for i in range(image_list.size().getInfo()):
                image = ee.Image(image_list.get(i))
                
                self.export_image(image, i)        
        except Exception as e:
            pass
            
            

class S1_download(object):
    
    def __init__(self, start_date, end_date, bands, ee_geometry, output_dir):
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.ee_geometry = ee_geometry
        self.output_dir = output_dir
            
    def lee_filter(self, image, kernel_size=3):
        """Applies a Lee filter to an image.
        
        Args:
            image: The input SAR image (single band).
            kernel_size: Size of the moving window (e.g., 3x3, 5x5).
        
        Returns:
            Filtered image.
        """
        # Compute mean and variance in the moving window
        mean = image.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(kernel_size / 2, units='pixels')
        )
        variance = image.reduceNeighborhood(
            reducer=ee.Reducer.variance(),
            kernel=ee.Kernel.square(kernel_size / 2, units='pixels')
        )
        # Coefficient of variation
        ratio = variance.divide(mean.pow(2))
        # Lee filter formula
        weight = ratio.divide(ratio.add(1))
        filtered = mean.add(weight.multiply(image.subtract(mean)))
        return filtered

    def apply_lee_filter_to_collection(self, collection, kernel_size=3):
        return collection.map(lambda img: self.lee_filter(img, kernel_size).copyProperties(img, img.propertyNames()))


    # Function to calculate monthly mean
    def calculate_monthly_mean(self, image_collection):
        start = ee.Date(self.start_date)
        end = ee.Date(self.end_date)
        months = ee.List.sequence(0, end.difference(start, 'month').getInfo())

        def calculate_mean_for_month(n):
            month_start = start.advance(n, 'month')
            month_end = month_start.advance(1, 'month')
            monthly_mean = (
                image_collection.filterDate(month_start, month_end)
                .mean()
                .set('month', month_start.format('YYYY-MM'))
            )
            return monthly_mean

        monthly_means = months.map(calculate_mean_for_month)
        return ee.ImageCollection(monthly_means)


    def process_S1(self):
        # Load the Sentinel-1 ImageCollection, filter to Jun-Sep 2020 observations
        sentinel_1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(
            self.start_date, self.end_date
        )

        # Filter the Sentinel-1 collection by metadata properties
        vv_vh_iw = (
            sentinel_1.filter(
                # Filter to get images with VV and VH dual polarization
                ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')
            )
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(
                # Filter to get images collected in interferometric wide swath mode
                ee.Filter.eq('instrumentMode', 'IW')
            )
        )
        
        # Separate ascending and descending orbit images into distinct collections
        vv_vh_iw_asc = vv_vh_iw.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        vv_vh_iw_desc = vv_vh_iw.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        
        return vv_vh_iw_asc, vv_vh_iw_desc
    
    def process_all_S1(self):
        
        vv_vh_iw_asc, vv_vh_iw_desc = self.process_S1()
    
        # Filtered ImageCollections
        vv_vh_iw_asc_filtered =   self.apply_lee_filter_to_collection(vv_vh_iw_asc.select('VH'))
        vv_vh_iw_desc_filtered =  self.apply_lee_filter_to_collection(vv_vh_iw_desc.select('VH'))
        vv_asc_dec_combined_filtered = self.apply_lee_filter_to_collection(vv_vh_iw_asc.merge(vv_vh_iw_desc).select('VV'))

        # Calculate monthly means with filtered data
        vh_iw_asc_monthly_means_filtered   = self.calculate_monthly_mean(vv_vh_iw_asc_filtered)
        vh_iw_desc_monthly_means_filtered  = self.calculate_monthly_mean(vv_vh_iw_desc_filtered)
        vv_asc_desc_monthly_means_filtered = self.calculate_monthly_mean(vv_asc_dec_combined_filtered)

        # Export results for filtered data
        self.export_monthly_means(vh_iw_asc_monthly_means_filtered,   'Filtered_VH_ASC')
        self.export_monthly_means(vh_iw_desc_monthly_means_filtered,  'Filtered_VH_DESC')
        self.export_monthly_means(vv_asc_desc_monthly_means_filtered, 'Filtered_VV_ASC_DESC')
        
    # Export each monthly mean
    def export_monthly_means(self, collection, prefix):
        collection_list = collection.toList(collection.size())
        for i in range(collection.size().getInfo()):
            image = ee.Image(collection_list.get(i))
            month = image.get('month').getInfo()
            year = image.get('year').getInfo()
            task = ee.batch.Export.image.toDrive(
                image=image.clip(self.ee_geometry),
                description=f'{prefix}_{year}_{month}',
                folder= self.output_dir,
                scale=10,
                region=self.eegeometry.getInfo()['coordinates'],
                crs='EPSG:4326'
            )
            task.start()
            print(f"Export started for {prefix} {month}!")            