import os
import cdsapi

# Create a directory to save the files
save_dir = 'data/vietnam_cloud_data'
os.makedirs(save_dir, exist_ok=True)

# Initialize the CDS API client
c = cdsapi.Client()

# List of selected years
years = ['2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2019', '2021', '2023']
# List of selected months
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# List of selected days
days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
# Time intervals for every 2 hours
times = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00']
# North, West, South, East coordinates for Vietnam
vietnam_area = [23, 102, 8, 110]

# Loop over each year and retrieve data
for year in years:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'total_cloud_cover', 'high_cloud_cover', 'medium_cloud_cover', 'low_cloud_cover',
                'total_column_cloud_ice_water', 'total_column_cloud_liquid_water'
            ],
            'year': year,
            'month': days,
            'day': days,
            'time': times,
            'area': vietnam_area,
        },
        os.path.join(save_dir, f'vietnam_cloud_data_{year}.nc')
    )
