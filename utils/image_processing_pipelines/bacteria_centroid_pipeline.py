from .image_pipeline import ImagePipeline

import cv2
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, label
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class BacteriaCentroidPipeline(ImagePipeline):
    """
    A pipeline for processing bacterial images to identify centroids and key points.
    This pipeline processes 3D tomographic data to:
    - Identify bacterial regions through intensity analysis
    - Find optimal slices using entropy calculations
    - Generate positive and negative points for segmentation
    - Handle image preprocessing and normalization
    """
    def __init__(
            self,
            downsample_factor:int=5,
            dark_group_percentile:int=10,
            entropy_slices_to_average:int=15,
            num_negative_points:int=3,
            num_positive_points:int=2,
            temp_image_database:str="temp_image_database",
        ):
        """
        Initialize the pipeline with configurable parameters.

        Args:
            downsample_factor: Factor to reduce image dimensions for faster processing
            dark_group_percentile: Percentile threshold for identifying dark regions
            entropy_slices_to_average: Number of adjacent slices to average for entropy
            num_negative_points: Number of background points to identify
            num_positive_points: Number of foreground points to identify
            temp_image_database: Directory path for temporary image storage
        """
        self.downsample_factor = downsample_factor
        self.dark_group_percentile = dark_group_percentile
        self.entropy_slices_to_average = entropy_slices_to_average
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.temp_image_database = temp_image_database
    
    def process_image(
            self,
            o_tomo,
            key,
            **kwargs
        ):
        """
        Main processing pipeline for bacterial image analysis.
        
        Processing Steps:
        1. Normalize and equalize the tomogram for consistent intensity
        2. Downsample data and identify dark connecting groups
        3. Find slice with highest entropy for optimal analysis
        4. Generate positive points (bacterial regions) and negative points (background)
        5. Scale coordinates for different resolution outputs
        6. Generate and save processed images
        7. Calculate entropy boundaries for sequence analysis
        
        Args:
            o_tomo: Original 3D tomogram data
            key: Unique identifier for this processing run
            **kwargs: Additional processing parameters
            
        Returns:
            dict: Contains processed points, file paths, and boundary values:
                - jpeg_feeding_points: Points scaled for JPEG output
                - np_feeding_points: Points in numpy array coordinates
                - stopper_points: Sequence boundary values
                - temp_bacteria_path: Path to temporary processed images
        """
        # Clean key by removing anything after a period
        clean_key = str(key).split('.')[0]
        temp_image_database = os.path.join("temporary_files", self.temp_image_database + "_" + clean_key)
        # print(f"Processing image with key: {key}")
        
        # Step 1: Normalize and equalize for consistent intensity range
        # print("Normalizing and equalizing the tomogram...")
        o_tomo = self.min_max_normalize(o_tomo) 
        o_tomo = self.histogram_equalization_3d(o_tomo)
        
        # Step 2: Reduce data size and identify dark regions
        # print("Downsampling the tomogram...")
        tomo = self.downsample_3d_average(o_tomo, self.downsample_factor)
        print("Selecting the highest ranked dark group...")
        binary_mask, mask, s_axis = self.select_ranked_dark_group(
            array_3d=tomo, 
            percentile=self.dark_group_percentile
        )

        # Step 3: Find optimal slice based on entropy
        # print("Finding the slice with the highest entropy...")
        mask_entropy_slice = self.max_entropy_slice(
            array=mask, 
            num_slices=self.entropy_slices_to_average
        )
        print(f"Entropy mask slice determined: {mask_entropy_slice}")
        
        # Step 4: Generate analysis points
        # print("Finding positive and negative points...")
        positive_points, negative_points = self.find_points(
            array_3d=mask, 
            slice_number=mask_entropy_slice, 
            num_negative_points=self.num_negative_points, 
            num_positive_points=self.num_positive_points, 
            min_distance_percent=0.1
        )
        # print(f"Positive points found: {positive_points}")
        # print(f"Negative points found: {negative_points}")
        
        # Step 5: Scale points back to original resolution
        entropy_slice = self.downsample_factor * mask_entropy_slice
        np_positive_points = self.upscale_points(
            scaling_factor=self.downsample_factor, 
            points=positive_points
        )
        np_negative_points = self.upscale_points(
            scaling_factor=self.downsample_factor, 
            points=negative_points
        )
        
        # Step 6: Generate and save processed images
        print("Generating JPEG images for analysis...")
        jpeg_shape = self.generate_images(o_tomo, save_path=temp_image_database)
        original_height, original_width = o_tomo.shape[1], o_tomo.shape[2]
        # print("Frame:", entropy_slice)

        # Scale points for JPEG output
        # print("Rescaling points for JPEG images...")
        jpeg_positive_points = self.rescale_coordinates(
            (original_height, original_width), 
            jpeg_shape, 
            np_positive_points
        )
        jpeg_negative_points = self.rescale_coordinates(
            (original_height, original_width), 
            jpeg_shape, 
            np_negative_points
        )
        # print(f"JPEG Positive Points: {jpeg_positive_points}")
        # print(f"JPEG Negative Points: {jpeg_negative_points}")

        # Organize points for output
        jpeg_feeding_points = {
            entropy_slice: {
                "obj_id": 1,
                "points": np.concatenate(
                    (jpeg_positive_points, jpeg_negative_points), 
                    axis=0
                ).astype(np.float32),
                "labels": np.array(
                    [1] * len(jpeg_positive_points) + [0] * len(jpeg_negative_points), 
                    dtype=np.int32
                )
            },
        }
        
        np_feeding_points = {
            entropy_slice: {
                "obj_id": 1,
                "points": np.concatenate(
                    (np_positive_points, np_negative_points), 
                    axis=0
                ).astype(np.float32),
                "labels": np.array(
                    [1] * len(np_positive_points) + [0] * len(np_negative_points), 
                    dtype=np.int32
                )
            },
        }

        # Calculate entropy values and determine sequence boundaries
        # print("Calculating entropy values across all slices...")
        entropy_list = self.calculate_entropy_slices(mask)
        
        # print("Determining slices below clustering threshold...")
        entropy_dict = self.slices_below_clustering(entropy_list)
        # print("Identifying sequence boundaries...")
        stopper_values = self.get_sequence_boundaries(entropy_dict)
        
        # Scale stopper values to original resolution
        stopper_values = {
            'first_sequence_max': stopper_values[0] * self.downsample_factor 
                if stopper_values[0] is not None else None,
            'second_sequence_min': stopper_values[1] * self.downsample_factor 
                if stopper_values[1] is not None else None
        }
        
        print("Processing complete. Returning results.")
        
        return {
            "jpeg_feeding_points": jpeg_feeding_points,
            "np_feeding_points": np_feeding_points,
            "stopper_points": stopper_values,
            "temp_bacteria_path": temp_image_database,
        }
    
    def downsample_3d_average(self, image_3d, factor):
        """
        Downsamples a 3D image by averaging non-overlapping blocks using vectorized operations.
        This reduces memory usage while preserving important features through averaging.

        Args:
            image_3d: 3D numpy array to downsample
            factor: Integer reduction factor (e.g., factor=8 reduces size by 1/8th)

        Returns:
            numpy.ndarray: Downsampled 3D array
        """
        # Calculate new dimensions ensuring clean division by factor
        new_shape = (
            image_3d.shape[0] // factor,
            image_3d.shape[1] // factor,
            image_3d.shape[2] // factor
        )
        
        # Reshape array to create blocks for averaging
        # This creates a view of the array grouped into factor-sized chunks
        reshaped = image_3d[:new_shape[0] * factor, 
                           :new_shape[1] * factor, 
                           :new_shape[2] * factor].reshape(
            new_shape[0], factor,
            new_shape[1], factor,
            new_shape[2], factor
        )
        
        # Average along factor dimensions to create downsampled array
        downsampled = reshaped.mean(axis=(1, 3, 5))
        
        return downsampled
    
    def select_ranked_dark_group(self, array_3d, percentile=5, rank=1, connectivity=2):
        """
        Identifies and ranks dark regions within a 3D array based on size and intensity.
        Focuses analysis on the middle 60% of x and y dimensions to avoid edge artifacts.

        Args:
            array_3d: Input 3D array
            percentile: Threshold percentile for dark value selection
            rank: Size-based rank of group to return (1 = largest)
            connectivity: Neighbor connectivity for grouping (1=face, 2=edge/corner)

        Returns:
            tuple: (ranked_group_mask, modified_array, shortest_axis_index)
        """
        # Normalize array to [0,1] range for consistent thresholding
        array_min = np.min(array_3d)
        array_max = np.max(array_3d)
        
        # Handle edge case where all values are identical
        if array_max - array_min != 0:
            normalized_array = (array_3d - array_min) / (array_max - array_min)
        else:
            normalized_array = np.zeros_like(array_3d)

        # Find shortest axis for optimal processing
        dims = sorted(enumerate(array_3d.shape), key=lambda x: x[1])
        z_index, z_size = dims[0]
        other_indices = [dims[1][0], dims[2][0]]

        # Get dimensions for other axes
        x_size = array_3d.shape[other_indices[0]]
        y_size = array_3d.shape[other_indices[1]]

        # Calculate boundaries for middle 60% region (20% margin on each side)
        x_trim_start = int(0.1 * x_size)
        x_trim_end = x_size - x_trim_start
        y_trim_start = int(0.1 * y_size)
        y_trim_end = y_size - y_trim_start

        # Create appropriate slice objects based on shortest axis
        if z_index == 0:
            restricted_region = (slice(None), 
                               slice(x_trim_start, x_trim_end), 
                               slice(y_trim_start, y_trim_end))
        elif z_index == 1:
            restricted_region = (slice(x_trim_start, x_trim_end), 
                               slice(None), 
                               slice(y_trim_start, y_trim_end))
        else:
            restricted_region = (slice(x_trim_start, x_trim_end), 
                               slice(y_trim_start, y_trim_end), 
                               slice(None))
        
        # Apply restriction and create dark values mask
        restricted_normalized_array = normalized_array[restricted_region]
        restricted_dark_values_mask = restricted_normalized_array < np.percentile(
            restricted_normalized_array, 
            percentile
        )
        
        # Label connected components in restricted region
        structure = np.ones((3, 3, 3)) if connectivity == 2 else None
        labeled_array, num_features = label(restricted_dark_values_mask, structure=structure)

        # Handle case where no dark groups are found
        if num_features == 0:
            print("No dark-value groups found.")
            return (np.zeros_like(array_3d), 
                   0, 
                   normalized_array, 
                   np.ones_like(array_3d))

        # Count size of each labeled group
        label_counts = np.bincount(labeled_array.flat)

        # Sort groups by size (excluding background label 0)
        sorted_labels_and_sizes = sorted(
            enumerate(label_counts[1:], start=1), 
            key=lambda x: x[1], 
            reverse=True
        )

        # Handle case where requested rank exceeds number of groups
        if rank > len(sorted_labels_and_sizes):
            print(f"Rank {rank} exceeds the number of detected groups. Returning empty mask.")
            return (np.zeros_like(array_3d), 
                   0, 
                   normalized_array, 
                   np.ones_like(array_3d))

        # Get label for requested rank
        ranked_group_label, ranked_group_size = sorted_labels_and_sizes[rank - 1]

        # Create mask for ranked group in restricted region
        ranked_group_mask_restricted = labeled_array == ranked_group_label

        # Expand mask to original array dimensions
        ranked_group_mask = np.zeros_like(normalized_array, dtype=bool)
        ranked_group_mask[restricted_region] = ranked_group_mask_restricted

        # Create modified array with ranked group preserved
        modified_array = np.where(ranked_group_mask, array_3d, 1)

        return ranked_group_mask, modified_array, z_index

    def max_entropy_slice(self, array, num_slices=10):
        """
        Identifies the slice with maximum entropy in a 3D array by analyzing averaged groups of slices.
        This helps find the most information-rich cross-section of the data.

        Args:
            array: 3D numpy array to analyze
            num_slices: Number of adjacent slices to average for smoother entropy calculation

        Returns:
            int: Index of the slice with maximum entropy
        """
        # Find shortest axis for consistent slice extraction
        shortest_axis = np.argmin(array.shape)
        
        max_entropy_value = -np.inf
        max_entropy_slice = -1

        # Iterate through slices along shortest axis
        for frame in range(array.shape[shortest_axis]):
            # Calculate slice range for averaging
            start_slice = max(0, frame - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], frame + num_slices // 2 + 1)
            
            # Extract and average slices based on axis orientation
            if shortest_axis == 0:
                slices = array[start_slice:end_slice, :, :]
                slice_ = np.mean(slices, axis=0)
            elif shortest_axis == 1:
                slices = array[:, start_slice:end_slice, :]
                slice_ = np.mean(slices, axis=1)
            elif shortest_axis == 2:
                slices = array[:, :, start_slice:end_slice]
                slice_ = np.mean(slices, axis=2)
            
            # Calculate entropy for averaged slice
            flattened_slice = slice_.flatten()
            hist = np.histogram(flattened_slice, bins=256)[0]
            entropy_value = entropy(hist, base=2)
            
            # Update maximum if current entropy is higher
            if entropy_value > max_entropy_value:
                max_entropy_value = entropy_value
                max_entropy_slice = frame

        return max_entropy_slice
    
    def find_points(self, array_3d, slice_number, num_slices=10, threshold=0.9, 
                    num_negative_points=5, num_positive_points=5, 
                    min_distance_percent=0.05, negative_min_distance_percent=0.1):
        """
        Identifies key points in the image: centroid of dark regions (positive points)
        and contrasting points in non-dark regions (negative points).

        Args:
            array_3d: Input 3D array
            slice_number: Target slice for point identification
            num_slices: Number of slices to average
            threshold: Threshold for identifying dark regions
            num_negative_points: Number of background points to identify
            num_positive_points: Number of foreground points to identify
            min_distance_percent: Minimum distance from dark regions (as percentage)
            negative_min_distance_percent: Minimum distance between negative points

        Returns:
            tuple: (positive_points, negative_points) Lists of coordinate tuples
        """
        # Find shortest axis for consistent slice extraction
        shortest_axis = np.argmin(array_3d.shape)
        slice_number = np.clip(slice_number, 0, array_3d.shape[shortest_axis] - 1)

        # Extract and average slices based on axis orientation
        if shortest_axis == 0:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[start_slice:end_slice, :, :]
            slice_2d = np.mean(slices, axis=0)
        elif shortest_axis == 1:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[:, start_slice:end_slice, :]
            slice_2d = np.mean(slices, axis=1)
        elif shortest_axis == 2:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array_3d.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array_3d[:, :, start_slice:end_slice]
            slice_2d = np.mean(slices, axis=2)
        
        # Create mask for dark regions
        black_mask = slice_2d < threshold

        # Find centroid of dark regions
        centroid = center_of_mass(black_mask)
        centroid = (centroid[1], centroid[0])  # Convert to (x,y) format

        # Define region for positive points (middle 40% of object)
        black_coords = np.column_stack(np.where(black_mask))
        min_y, min_x = black_coords.min(axis=0)
        max_y, max_x = black_coords.max(axis=0)

        # Calculate margins for middle 40% region
        x_margin = 0.2 * (max_x - min_x)  # 20% margin on each side
        y_margin = 0.2 * (max_y - min_y)

        # Create mask for middle 40% region
        middle_40_mask = np.zeros_like(black_mask, dtype=bool)
        middle_40_mask[int(min_y + y_margin):int(max_y - y_margin), 
                      int(min_x + x_margin):int(max_x - x_margin)] = True

        # Combine masks to get valid positive point regions
        valid_positive_mask = middle_40_mask & black_mask
        positive_coords = np.column_stack(np.where(valid_positive_mask))

        # Adjust number of points if fewer are available
        if len(positive_coords) < num_positive_points:
            num_positive_points = len(positive_coords)

        # Sample positive points randomly from valid regions
        if num_positive_points > 0:
            positive_points = positive_coords[
                np.random.choice(positive_coords.shape[0], 
                               num_positive_points, 
                               replace=False)
            ]
        else:
            positive_points = np.array([])

        # Add centroid to positive points
        positive_points = np.vstack([
            np.array([[int(centroid[1]), int(centroid[0])]]), 
            positive_points
        ])

        # Convert coordinates to (x,y) format
        positive_points = [(pt[1], pt[0]) for pt in positive_points]

        # Find negative points in non-dark regions
        non_black_mask = ~black_mask

        # Define middle 60% region for negative points
        x_start = int(black_mask.shape[1] * 0.2)
        x_end = int(black_mask.shape[1] * 0.8)
        y_start = int(black_mask.shape[0] * 0.2)
        y_end = int(black_mask.shape[0] * 0.8)

        # Create mask for middle 60% region
        middle_60_mask = np.zeros_like(non_black_mask, dtype=bool)
        middle_60_mask[y_start:y_end, x_start:x_end] = True

        # Combine masks for valid negative point regions
        candidate_mask = non_black_mask & middle_60_mask

        if not candidate_mask.any():
            return positive_points, None

        # Calculate minimum distances for point placement
        avg_dimension_size = (black_mask.shape[1] + black_mask.shape[0]) / 2
        min_distance_pixels = avg_dimension_size * min_distance_percent
        negative_min_distance_pixels = avg_dimension_size * negative_min_distance_percent

        # Calculate distance transform for spacing constraints
        distances = distance_transform_edt(non_black_mask)
        valid_distance_mask = distances >= min_distance_pixels
        final_candidate_mask = candidate_mask & valid_distance_mask

        if not final_candidate_mask.any():
            return positive_points, None

        # Get coordinates of valid negative points
        negative_coords = np.column_stack(np.where(final_candidate_mask))

        # Adjust number of points if fewer are available
        if len(negative_coords) < num_negative_points:
            num_negative_points = len(negative_coords)

        # Initialize negative points list
        selected_negative_points = []

        # Sample negative points ensuring minimum distance between them
        np.random.shuffle(negative_coords)
        for candidate in negative_coords:
            if len(selected_negative_points) == 0:
                selected_negative_points.append(candidate)
            else:
                # Check distance to already selected points
                distances_to_selected = cdist([candidate], selected_negative_points)
                if np.all(distances_to_selected >= negative_min_distance_pixels):
                    selected_negative_points.append(candidate)
                    if len(selected_negative_points) >= num_negative_points:
                        break

        # Convert coordinates to (x,y) format
        return positive_points, [(pt[1], pt[0]) for pt in selected_negative_points]

    def upscale_points(self, scaling_factor, points):
        """
        Scales up point coordinates by a given factor. Handles both single points
        and lists of points for flexible coordinate transformation.

        Args:
            scaling_factor: Factor to multiply coordinates by
            points: Single point (x,y) or list of points to upscale

        Returns:
            tuple or list: Upscaled point(s) coordinates

        Raises:
            ValueError: If input points format is invalid
        """
        points_array = np.array(points)

        # Handle single point case
        if points_array.ndim == 1 and points_array.size == 2:
            upscaled_point = points_array * scaling_factor
            return tuple(upscaled_point)
        # Handle multiple points case
        elif points_array.ndim == 2 and points_array.shape[1] == 2:
            upscaled_points = points_array * scaling_factor
            return [tuple(point) for point in upscaled_points]
        else:
            raise ValueError("Input must be either a single 2D point (x,y) or a list of 2D points.")

    def rescale_coordinates(self, original_shape, new_shape, points):
        """
        Rescales point coordinates from original image dimensions to new dimensions.
        Useful when converting between different resolution outputs.

        Args:
            original_shape: Tuple of (height, width) for original image
            new_shape: Tuple of (height, width) for target image
            points: List of (x,y) coordinates to rescale

        Returns:
            numpy.ndarray: Rescaled coordinates
        """
        if not points:
            return np.array([])
            
        points_array = np.array(points)
        
        # Calculate scaling factors
        y_scale = new_shape[0] / original_shape[0]
        x_scale = new_shape[1] / original_shape[1]
        
        # Apply scaling to coordinates
        scaled_points = points_array * np.array([x_scale, y_scale])
        
        return scaled_points

    def generate_images(self, array, save_path, max_frame_size=(512, 512), 
                       reflect_edges=False, num_slices=10):
        """
        Generates and saves JPEG images from 3D array slices with preprocessing options.
        Includes options for size limits, edge reflection, and slice averaging.

        Args:
            array: 3D numpy array to process
            save_path: Directory to save generated images
            max_frame_size: Maximum dimensions for output images
            reflect_edges: Whether to reflect array at edges
            num_slices: Number of slices to average

        Returns:
            tuple: (height, width) of resized frames
        """
        # Find shortest axis for consistent slicing
        shortest_axis = np.argmin(array.shape)
        os.makedirs(save_path, exist_ok=True)

        # Determine frame dimensions based on slicing axis
        if shortest_axis == 0:
            height, width = array.shape[1], array.shape[2]
        elif shortest_axis == 1:
            height, width = array.shape[0], array.shape[2]
        else:
            height, width = array.shape[0], array.shape[1]
        
        # Calculate resize dimensions if needed
        if height > max_frame_size[1] or width > max_frame_size[0]:
            resize_factor_h = max_frame_size[1] / height
            resize_factor_w = max_frame_size[0] / width
            resize_factor = min(resize_factor_h, resize_factor_w)
            height = int(height * resize_factor)
            width = int(width * resize_factor)
        
        resized_shape = (height, width)
        
        # Add padding if reflection is enabled
        if reflect_edges:
            pad_width = [(num_slices // 2, num_slices // 2) if i == shortest_axis 
                        else (0, 0) for i in range(3)]
            array = np.pad(array, pad_width, mode='reflect')

        # Process and save each slice
        for i in range(array.shape[shortest_axis] - (num_slices if reflect_edges else 0)):
            # Calculate slice range for averaging
            start_idx = max(0, i - num_slices // 2)
            end_idx = min(array.shape[shortest_axis], i + num_slices // 2 + 1)
            
            # Extract and average slices based on axis
            if shortest_axis == 0:
                slice_ = np.mean(array[start_idx:end_idx, :, :], axis=0)
            elif shortest_axis == 1:
                slice_ = np.mean(array[:, start_idx:end_idx, :], axis=1)
            else:
                slice_ = np.mean(array[:, :, start_idx:end_idx], axis=2)
            
            # Normalize to 8-bit range
            slice_normalized = cv2.normalize(
                slice_, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Resize if necessary
            if height != slice_normalized.shape[0] or width != slice_normalized.shape[1]:
                slice_resized = cv2.resize(
                    slice_normalized, (width, height), 
                    interpolation=cv2.INTER_AREA
                )
            else:
                slice_resized = slice_normalized
            
            # Convert to BGR for JPEG saving
            slice_colored = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
            slice_flipped = cv2.flip(slice_colored, 0)
            
            # Save frame
            frame_path = os.path.join(save_path, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, slice_flipped)

        return resized_shape
            
    def calculate_entropy_slices(self, arr):
        """
        Calculates entropy values for each slice in the 3D array, using a sliding window
        approach to average neighboring slices for smoother results.

        Args:
            arr: 3D numpy array to analyze

        Returns:
            dict: Mapping of slice indices to their entropy values
        """
        # Find shortest axis for consistent slice analysis
        shortest_axis = np.argmin(arr.shape)
        entropy_dict = {}
        num_slices = arr.shape[shortest_axis]
        neighbors = 10  # Window size for averaging
        
        def calculate_entropy(slice_):
            """
            Helper function to calculate Shannon entropy of a single slice.
            
            Args:
                slice_: 2D numpy array
            Returns:
                float: Entropy value
            """
            values, counts = np.unique(slice_.flatten(), return_counts=True)
            probs = counts / counts.sum()
            return entropy(probs)
        
        # Calculate entropy for each slice using sliding window
        for i in range(num_slices):
            # Define window boundaries
            start_idx = max(0, i - neighbors // 2)
            end_idx = min(num_slices, i + neighbors // 2 + 1)
            
            # Extract and average slices based on axis orientation
            if shortest_axis == 0:
                averaged_slice = np.mean(arr[start_idx:end_idx, :, :], axis=0)
            elif shortest_axis == 1:
                averaged_slice = np.mean(arr[:, start_idx:end_idx, :], axis=1)
            else:
                averaged_slice = np.mean(arr[:, :, start_idx:end_idx], axis=2)
            
            entropy_dict[i] = calculate_entropy(averaged_slice)
        
        return entropy_dict

    def slices_below_clustering(self, entropy_dict, n_clusters=2):
        """
        Uses K-means clustering to identify slices with low entropy values.
        This helps identify regions of interest in the volume.

        Args:
            entropy_dict: Dictionary mapping slice indices to entropy values
            n_clusters: Number of clusters for K-means analysis

        Returns:
            dict: Filtered dictionary containing only low-entropy slices
        """
        # Convert entropy values to array for clustering
        entropy_values = np.array(list(entropy_dict.values())).reshape(-1, 1)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(entropy_values)
        
        # Find cluster with lowest mean entropy
        cluster_means = [np.mean(entropy_values[labels == i]) for i in range(n_clusters)]
        low_entropy_cluster = np.argmin(cluster_means)
        
        # Filter slices belonging to low entropy cluster
        slices_below_threshold = {
            slice_num: entropy 
            for slice_num, entropy in entropy_dict.items()
            if labels[list(entropy_dict.keys()).index(slice_num)] == low_entropy_cluster
        }
        
        return slices_below_threshold

    def get_sequence_boundaries(self, entropy_dict):
        """
        Identifies boundaries of consecutive sequences in entropy data.
        Used to determine where significant changes in entropy occur.

        Args:
            entropy_dict: Dictionary of entropy values by slice index

        Returns:
            tuple: (first_sequence_max, second_sequence_min) Boundary indices
                  None values indicate no boundary found
        """
        # Sort slice indices for sequential analysis
        sorted_keys = sorted(entropy_dict.keys())
        sequences = []
        current_sequence = [sorted_keys[0]]
        
        # Group consecutive indices into sequences
        for i in range(1, len(sorted_keys)):
            if sorted_keys[i] == sorted_keys[i - 1] + 1:
                current_sequence.append(sorted_keys[i])
            else:
                sequences.append(current_sequence)
                current_sequence = [sorted_keys[i]]
        
        # Add final sequence if exists
        if current_sequence:
            sequences.append(current_sequence)

        # Initialize boundary values
        first_sequence_max = None
        second_sequence_min = None

        if sequences:
            # Determine sequence type based on starting index
            if sequences[0][0] <= 10:  # First sequence starts near beginning
                first_sequence_max = sequences[0][-1]
            else:  # First sequence starts later
                second_sequence_min = sequences[0][0]

            # Handle second sequence if it exists
            if len(sequences) > 1:
                if second_sequence_min is None:
                    second_sequence_min = sequences[1][0]
                else:
                    second_sequence_min = sequences[1][0]

        return first_sequence_max, second_sequence_min

    def min_max_normalize(self, array):
        """
        Normalizes array values to range [0,1] using min-max normalization.
        Handles edge case where all values are identical.

        Args:
            array: Input numpy array

        Returns:
            numpy.ndarray: Normalized array with values in [0,1] range
        """
        array = array.astype(np.float32)
        min_val = np.min(array)
        max_val = np.max(array)
        
        # Handle case where all values are identical
        if min_val == max_val:
            return np.zeros(array.shape, dtype=np.float32)
        
        return (array - min_val) / (max_val - min_val)

    def histogram_equalization_3d(self, image):
        """
        Applies histogram equalization to 3D array to enhance contrast.
        Uses cumulative distribution function for intensity mapping.

        Args:
            image: 3D numpy array

        Returns:
            numpy.ndarray: Contrast-enhanced array through histogram equalization
        """
        # Calculate histogram with 256 bins
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 1])

        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        # Normalize CDF to [0,1] range
        cdf_normalized = cdf / cdf.max()

        # Apply equalization through linear interpolation
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        
        return image_equalized.reshape(image.shape)
    
    def save_specific_slice_centroid(self, array, slice_number, num_slices=10, 
                                   custom_name=None, save_path=None, 
                                   positive_points=None, negative_points=None):
        """
        Debugging/visualization function to save a specific slice with marked points.
        Useful for verifying point detection and placement.

        Args:
            array: 3D numpy array to visualize
            slice_number: Specific slice to plot
            num_slices: Number of slices to average
            custom_name: Custom title for plot
            save_path: Path to save visualization
            positive_points: List of positive points to mark (red)
            negative_points: List of negative points to mark (blue)
        """
        # Find shortest axis for consistent slicing
        shortest_axis = np.argmin(array.shape)
        slice_number = np.clip(slice_number, 0, array.shape[shortest_axis] - 1)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Extract and average slices based on axis
        if shortest_axis == 0:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array[start_slice:end_slice, :, :]
            slice_ = np.mean(slices, axis=0)
            xlabel, ylabel = 'x', 'y'
        elif shortest_axis == 1:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array[:, start_slice:end_slice, :]
            slice_ = np.mean(slices, axis=1)
            xlabel, ylabel = 'z', 'y'
        else:
            start_slice = max(0, slice_number - num_slices // 2)
            end_slice = min(array.shape[shortest_axis], slice_number + num_slices // 2 + 1)
            slices = array[:, :, start_slice:end_slice]
            slice_ = np.mean(slices, axis=2)
            xlabel, ylabel = 'z', 'x'
        
        # Calculate slice statistics
        flattened_slice = slice_.flatten()
        slice_entropy = entropy(np.histogram(flattened_slice, bins=256)[0])
        mean_value = np.mean(slice_)
        
        # Create visualization
        im = ax.imshow(slice_, cmap='gray', origin='lower')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Plot points if provided
        if positive_points is not None:
            for point in positive_points:
                px, py = point
                ax.plot(px, py, 'ro', markersize=6, label="Positive Point")
        
        if negative_points is not None:
            for point in negative_points:
                px, py = point
                ax.plot(px, py, 'bo', markersize=6, label="Negative Point")
        
        # Set title and add statistics
        plot_title = custom_name if custom_name else f'Axis {shortest_axis} Slice {slice_number}'
        ax.set_title(plot_title)
        ax.text(0.05, 0.95, f'Entropy: {slice_entropy:.5f}', 
                transform=ax.transAxes, color='white', fontsize=10, 
                verticalalignment='top')
        ax.text(0.05, 0.90, f'Mean: {mean_value:.5f}', 
                transform=ax.transAxes, color='white', fontsize=10, 
                verticalalignment='top')
        
        # Save visualization if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            axis_names = ['z', 'x', 'y']
            base_name = f'slice_axis_{axis_names[shortest_axis]}_slice_{slice_number}'
            if custom_name:
                base_name += f'_{custom_name}'
            image_filename = os.path.join(save_path, f'{base_name}.png')
            plt.savefig(image_filename, dpi=100)
            print(f"Saved visualization to {image_filename}")
        
        plt.close(fig)
        
        
        
        
        