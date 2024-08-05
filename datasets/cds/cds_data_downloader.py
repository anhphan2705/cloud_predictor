import os
import cdsapi

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview

# Create a directory to save the files
save_dir = 'data/vietnam_temp_data'
# List of selected years
years = [year for year in range(2000, 2025)]
# List of selected months
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# List of selected days
days = [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
            '31'
        ]
# Time intervals for every 1 hours
times = [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ]
# List of variables of interest
variables = [
    '2m_temperature'
]
# lat_x, long_x, lat_y, long_y coordinates for Vietnam
vietnam_area = [23, 102, 8, 110]

# Make saving directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize the CDS API client
c = cdsapi.Client()

# Loop over each year and retrieve data
for year in years:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': year,
            'month': months,
            'day': days,
            'time': times,
            'area': vietnam_area,
        },
        os.path.join(save_dir, f'vietnam_cloud_data_{year}.nc')
    )