"""
Elevation Data Reader

Reads elevation data from etopo_land.png image where:
- White pixels = ocean (water)
- Yellow tones = land elevation (darker = higher, lighter = lower)
- Elevation range: 0m (lowest land) to 8000m (highest peak)
"""

import numpy as np
from PIL import Image
import os


class ElevationReader:
    """Read and query elevation data from image."""
    
    def __init__(self, image_path: str):
        """
        Initialize elevation reader.
        Pre-processes entire image to find min/max Luma values for fast queries.
        
        Args:
            image_path: Path to etopo_land.png image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Loading elevation data from {image_path}...")
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
        
        # Convert to numpy array for faster processing
        self.data = np.array(self.image)
        
        print(f"Image loaded: {self.width}x{self.height} pixels")
        print(f"Image mode: {self.image.mode}")
        
        # Detect if image is RGB or grayscale
        if len(self.data.shape) == 3:
            self.is_rgb = True
        else:
            self.is_rgb = False
        
        # Pre-process entire image to find min/max Luma values
        print("Pre-processing image to find elevation range...")
        self._preprocess_image()
    
    def _calculate_luma(self, pixel):
        """
        Calculate Luma (perceived brightness) from pixel.
        Uses standard formula: Y = 0.299*R + 0.587*G + 0.114*B
        
        Args:
            pixel: RGB tuple or grayscale value
            
        Returns:
            float: Luma value (0-255)
        """
        if self.is_rgb:
            r, g, b = float(pixel[0]), float(pixel[1]), float(pixel[2])
            return 0.299 * r + 0.587 * g + 0.114 * b
        else:
            return float(pixel)
    
    def _preprocess_image(self):
        """
        Pre-process entire image to find min/max Luma values for land pixels.
        This allows fast linear interpolation for queries.
        """
        luma_values = []
        
        # Scan entire image
        for y in range(self.height):
            for x in range(self.width):
                pixel = self.data[y, x]
                
                # Skip water pixels (white or near-white)
                if self._is_water_pixel(pixel):
                    continue
                
                # Calculate and store Luma for land pixels
                luma = self._calculate_luma(pixel)
                luma_values.append(luma)
        
        if not luma_values:
            raise ValueError("No land pixels found in image!")
        
        # Find min and max Luma values
        self.min_luma = np.min(luma_values)
        self.max_luma = np.max(luma_values)
        
        print(f"Luma range found: {self.min_luma:.2f} (highest) to {self.max_luma:.2f} (lowest)")
        print(f"Total land pixels analyzed: {len(luma_values)}")
        print(f"Pre-processing complete!")
    
    def _is_water_pixel(self, pixel):
        """
        Determine if a pixel represents water (ocean).
        White or very light pixels are considered water.
        
        Args:
            pixel: RGB tuple or grayscale value
            
        Returns:
            bool: True if pixel is water
        """
        if self.is_rgb:
            r, g, b = pixel[0], pixel[1], pixel[2]
            # White is (255, 255, 255), also consider very light colors as water
            # Water should be white or near-white
            return (r > 240 and g > 240 and b > 240)
        else:
            # Grayscale: white is 255
            return pixel > 240
    
    def _pixel_to_elevation(self, pixel):
        """
        Convert pixel color to elevation in meters using linear interpolation.
        Uses pre-calculated min/max Luma values for fast conversion.
        
        Args:
            pixel: RGB tuple or grayscale value
            
        Returns:
            float: Elevation in meters (0 to 8000), or None if water
        """
        if self._is_water_pixel(pixel):
            return None
        
        # Calculate Luma for this pixel
        luma = self._calculate_luma(pixel)
        
        # Linear interpolation:
        # min_luma (darkest) -> 8000m (highest)
        # max_luma (lightest) -> 0m (lowest)
        # Note: darker pixels have LOWER luma values
        
        if self.max_luma == self.min_luma:
            # Edge case: all land pixels have same brightness
            return 4000.0  # Return middle value
        
        # Normalize luma to 0-1 range (inverted: dark=1, light=0)
        normalized = (self.max_luma - luma) / (self.max_luma - self.min_luma)
        
        # Scale to 0-8000m
        elevation = normalized * 8000.0
        
        return elevation
    
    def get_region_info(self, x1, y1, x2, y2):
        """
        Get elevation statistics for a rectangular region.
        
        Args:
            x1, y1: Top-left corner (pixel coordinates)
            x2, y2: Bottom-right corner (pixel coordinates)
            
        Returns:
            dict: {
                'is_water': bool,  # True if region is mostly water
                'water_percentage': float,  # Percentage of water pixels
                'mean_elevation': float or None,  # Mean elevation in meters (None if all water)
                'min_elevation': float or None,
                'max_elevation': float or None,
                'land_pixels': int,  # Number of land pixels
                'water_pixels': int  # Number of water pixels
            }
        """
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Extract region
        region = self.data[y1:y2+1, x1:x2+1]
        
        elevations = []
        water_count = 0
        land_count = 0
        
        # Process each pixel in region
        for row in region:
            for pixel in row:
                if self._is_water_pixel(pixel):
                    water_count += 1
                else:
                    elevation = self._pixel_to_elevation(pixel)
                    if elevation is not None:
                        elevations.append(elevation)
                        land_count += 1
        
        total_pixels = water_count + land_count
        water_percentage = (water_count / total_pixels * 100) if total_pixels > 0 else 0
        
        # Region is considered water if > 50% is water
        is_water = water_percentage > 50.0
        
        result = {
            'is_water': is_water,
            'water_percentage': water_percentage,
            'land_pixels': land_count,
            'water_pixels': water_count,
            'mean_elevation': None,
            'min_elevation': None,
            'max_elevation': None
        }
        
        if elevations:
            result['mean_elevation'] = np.mean(elevations)
            result['min_elevation'] = np.min(elevations)
            result['max_elevation'] = np.max(elevations)
        
        return result
    
    def get_elevation_at_point(self, lat, lon):
        """
        Get elevation at a specific lat/lon point.
        Fast query using pre-calculated min/max Luma values.
        
        Args:
            lat: Latitude in degrees (-90 to +90)
            lon: Longitude in degrees (-180 to +180)
            
        Returns:
            float or None: Elevation in meters (0-8000), or None if water
        """
        # Convert lat/lon to pixel coordinates
        def lat_to_y(lat):
            normalized = (90.0 - lat) / 180.0
            return int(np.clip(normalized * (self.height - 1), 0, self.height - 1))
        
        def lon_to_x(lon):
            normalized = (lon + 180.0) / 360.0
            return int(np.clip(normalized * (self.width - 1), 0, self.width - 1))
        
        x = lon_to_x(lon)
        y = lat_to_y(lat)
        
        pixel = self.data[y, x]
        return self._pixel_to_elevation(pixel)
    
    def get_lat_lon_region_info(self, lat1, lon1, lat2, lon2):
        """
        Get elevation statistics for a region specified by latitude/longitude.
        
        Assumes image covers full globe:
        - Latitude: -90 (bottom) to +90 (top)
        - Longitude: -180 (left) to +180 (right)
        
        Args:
            lat1, lon1: First corner (degrees)
            lat2, lon2: Second corner (degrees)
            
        Returns:
            dict: Same as get_region_info()
        """
        # Convert lat/lon to pixel coordinates
        # Latitude: -90 to +90 maps to height-1 to 0 (inverted)
        # Longitude: -180 to +180 maps to 0 to width-1
        
        def lat_to_y(lat):
            # Normalize lat from [-90, 90] to [0, 1]
            normalized = (90.0 - lat) / 180.0
            return int(normalized * (self.height - 1))
        
        def lon_to_x(lon):
            # Normalize lon from [-180, 180] to [0, 1]
            normalized = (lon + 180.0) / 360.0
            return int(normalized * (self.width - 1))
        
        x1 = lon_to_x(lon1)
        y1 = lat_to_y(lat1)
        x2 = lon_to_x(lon2)
        y2 = lat_to_y(lat2)
        
        return self.get_region_info(x1, y1, x2, y2)
    
    def print_region_info(self, info):
        """Pretty print region information."""
        print("\n" + "="*50)
        print("Region Analysis:")
        print("="*50)
        
        if info['is_water']:
            print("Type: WATER (Ocean)")
        else:
            print("Type: LAND")
        
        print(f"Water coverage: {info['water_percentage']:.1f}%")
        print(f"Land pixels: {info['land_pixels']}")
        print(f"Water pixels: {info['water_pixels']}")
        
        if info['mean_elevation'] is not None:
            print(f"\nElevation Statistics:")
            print(f"  Mean: {info['mean_elevation']:.1f} m")
            print(f"  Min:  {info['min_elevation']:.1f} m")
            print(f"  Max:  {info['max_elevation']:.1f} m")
        else:
            print("\nNo elevation data (all water)")
        
        print("="*50)


def main():
    """Example usage."""
    # Path to elevation image
    image_path = "etopo_land.png"
    
    try:
        reader = ElevationReader(image_path)
        
        print("\n" + "="*50)
        print("Elevation Data Reader - Examples")
        print("="*50)
        
        # Example 1: Query a pixel region (x, y coordinates)
        print("\n1. Example: Small region (pixel coordinates)")
        info = reader.get_region_info(100, 100, 200, 200)
        reader.print_region_info(info)
        
        # Example 2: Query by latitude/longitude
        print("\n2. Example: Himalayas region (lat/lon)")
        # Himalayas: roughly 27-30°N, 85-90°E
        info = reader.get_lat_lon_region_info(27, 85, 30, 90)
        reader.print_region_info(info)
        
        # Example 3: Pacific Ocean
        print("\n3. Example: Pacific Ocean (lat/lon)")
        info = reader.get_lat_lon_region_info(-10, -150, 10, -130)
        reader.print_region_info(info)
        
        # Example 4: Amazon region
        print("\n4. Example: Amazon region (lat/lon)")
        info = reader.get_lat_lon_region_info(-5, -70, 0, -60)
        reader.print_region_info(info)
        
        print("\n" + "="*50)
        print("Usage in your code:")
        print("="*50)
        print("reader = ElevationReader('etopo_land.png')")
        print("info = reader.get_lat_lon_region_info(lat1, lon1, lat2, lon2)")
        print("if info['is_water']:")
        print("    print('This is ocean')")
        print("else:")
        print("    print(f'Mean elevation: {info[\"mean_elevation\"]:.1f} m')")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure 'etopo_land.png' is in the current directory.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
