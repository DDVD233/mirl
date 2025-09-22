import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import json
import os
from scipy.stats import gaussian_kde
import matplotlib.colors as colors

# Configure matplotlib to use Computer Modern font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

locations = [
    "Pittsburgh, PA, USA",  # CMU-MOSEI
    "Singapore, Singapore",  # MELD
    "Toronto, Canada",  # TESS
    "Philadelphia, PA, USA",  # CREMA-D
    "Beijing, China",  # CH-SIMSv2
    "Pittsburgh, PA, USA",  # SocialIQ2
    "Xi'an, China",  # IntentQA
    "Cambridge, MA, USA",  # MimeQA
    "Rochester, NY, USA",  # UR-FUNNYv2
    "Ann Arbor, MI, USA",  # MUStARD
    "Los Angeles, CA, USA",  # DAIC-WOZ
    "Guangdong Province, China",  # MMPsy
    "Paris, France",  # PTSD-in-the-Wild
]

CACHE_FILE = 'location_cache.json'


def load_cache():
    """Load the coordinate cache from file"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save the coordinate cache to file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_coordinates(location_name, cache):
    """Get coordinates for a location using Nominatim geocoder with caching"""
    # Check cache first
    if location_name in cache:
        return cache[location_name]

    geolocator = Nominatim(user_agent="clinical_data_map")
    try:
        # Add delay to respect Nominatim's usage policy
        time.sleep(1)
        location = geolocator.geocode(location_name)
        if location:
            coords = [location.latitude, location.longitude]
            # Only cache successful results
            cache[location_name] = coords
            save_cache(cache)
            return coords
        else:
            # try to search based on the city
            city = location_name.split(",")[-2].strip()
            location = geolocator.geocode(city)
            if location:
                coords = [location.latitude, location.longitude]
                # Only cache successful results
                cache[location_name] = coords
                save_cache(cache)
                return coords
            else:
                return None
    except GeocoderTimedOut:
        return None


def main():
    # Load the coordinate cache
    cache = load_cache()

    # Get coordinates for all locations
    coordinates = []
    for loc in locations:
        coords = get_coordinates(loc, cache)
        if coords:
            coordinates.append(coords)
        print(f"Processed: {loc} with coordinates {coords}")

    # Convert coordinates to numpy arrays for plotting
    coordinates = np.array(coordinates)
    lats = coordinates[:, 0]
    lons = coordinates[:, 1]

    # Create figure and projection
    plt.figure(figsize=(20, 10))
    # ax = plt.axes(projection=ccrs.Robinson())
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Create grid for kernel density estimation
    lon_grid = np.linspace(-180, 180, 361)  # One more point to match dimensions
    lat_grid = np.linspace(-90, 90, 181)  # One more point to match dimensions
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Stack coordinates for KDE
    xy_coords = np.vstack([lons, lats])

    # Perform kernel density estimation
    kde = gaussian_kde(xy_coords, bw_method=0.3)  # Adjust bandwidth as needed

    # Calculate density on the grid
    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    z = np.reshape(kde(positions).T, lon_mesh.shape)
    z = np.power(z, 0.2)

    land_mask = np.zeros(z.shape, dtype=bool)
    for lat_idx in range(len(lat_grid)):
        for lon_idx in range(len(lon_grid)):
            point = ccrs.PlateCarree().transform_point(lon_grid[lon_idx],
                                                       lat_grid[lat_idx],
                                                       ccrs.PlateCarree())
            # Check if point is on land
            for geom in cfeature.LAND.geometries():
                if geom.contains(shapely.geometry.Point(point)):
                    land_mask[lat_idx, lon_idx] = True
                    break
    z = np.where(land_mask, z, 0)

    # Plot heatmap
    # im = ax.pcolormesh(lon_grid, lat_grid, z, transform=ccrs.PlateCarree(),
    #                    cmap='YlOrRd', alpha=0.7, shading='nearest')
    # colors_list = ['blue', 'yellow', 'green']
    # n_bins = 256  # Number of color gradations
    # custom_cmap = colors.LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)

    # Use it in your plot
    # im = ax.pcolormesh(lon_grid, lat_grid, z, transform=ccrs.PlateCarree(),
    #                    cmap=custom_cmap, alpha=0.7, shading='nearest')
    im = ax.pcolormesh(lon_grid, lat_grid, z, transform=ccrs.PlateCarree(),
                       cmap='coolwarm', alpha=0.7, shading='nearest')

    # Plot points
    scatter = ax.scatter(lons, lats, c='white', s=50, transform=ccrs.PlateCarree(),
                         edgecolor='black', linewidth=1, alpha=0.7,
                         label='Source of Datasets')

    # Add colorbar
    # cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
    #                     label='Density of Data Collection Sites')

    # Add title and legend
    # plt.title('Global Distribution of Clinical Data Collection Sites',
    #           fontsize=16, pad=20)
    plt.legend(fontsize=34, loc='upper right')

    # Save the figure
    plt.savefig('data_locations_heatmap.jpg',
                dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()