from utils.image_processing_pipelines.image_pipeline import ImagePipeline
import numpy as np
from analyzer import Analysis
import os
import cv2
import torch
import warnings
import onnxruntime
import mrcfile
from .sam2.sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
import pickle
import matplotlib.pyplot as plt
from PIL import Image


class MembraneSegmenter(Analysis):
    """
    A class for segmenting membranes in tomographic data using SAM2 (Segment Anything Model 2).
    This class processes tomographic data through an image pipeline, applies SAM2 for segmentation,
    and saves the results as both MRC and pickle files.

    Attributes:
        ipp (ImagePipeline): Pipeline for processing images
        sam2_checkpoint (str): Path to SAM2 model checkpoint
        model_cfg (str): Path to model configuration file
        device (str): Computing device to use (cuda/cpu/mps)
        results_dir (str): Directory to save results
    """
    
    def __init__(
        self,
        image_processing_pipeline: ImagePipeline,
        sam2_checkpoint = "/sam2/checkpoints/sam2.1_hiera_large.pt",
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device="cuda",
        results_dir:str="./results"
    ):
        """
        Initialize the MembraneSegmenter with required parameters.
        
        Args:
            image_processing_pipeline: Pipeline for processing images
            sam2_checkpoint: Path to SAM2 model checkpoint
            model_cfg: Path to model configuration file
            device: Computing device to use (cuda/cpu/mps)
            results_dir: Directory to save results
        """
        self.ipp = image_processing_pipeline
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self.results_dir = results_dir
        
        # Add preloading of model
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="sam2/sam2")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
    def analyze(
            self,
            data_path,
            key,
            results: dict,
            **kwargs
        ):
        """
        Main analysis method to segment membranes in tomographic data.
        This method orchestrates the entire segmentation pipeline:
        1. Loads tomogram data
        2. Processes images through the pipeline
        3. Initializes the appropriate computing device
        4. Runs SAM2 segmentation
        5. Clears annotations outside the specified range
        6. Saves results in both pickle and MRC formats
        
        Args:
            data_path (str): Path to input tomogram data
            key (str): Identifier for the current analysis (typically filename)
            results (dict): Dictionary to store analysis results
            **kwargs: Additional keyword arguments
        
        Notes:
            - Results are saved in both .pkl and .mrc formats
            - The .pkl file contains both point data and masks
            - The .mrc file contains the 3D annotation array
        """
        print(f"Starting analysis for key: {key}")

        # Get TOMOGRAM FROM PATH
        print("Loading tomogram from path...")
        data = self.get_tomogram(data_path)
        
        # Process Images
        print("Processing images through the pipeline...")
        processed_data = self.ipp.process_image(data, key)
        
        # Initialize device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                  "give numerically different outputs and sometimes degraded performance on MPS.")

        # Load stack of JPEG images
        video_dir = processed_data["temp_bacteria_path"]
        print(f"Video directory: {video_dir}")
        feeding_points = processed_data["jpeg_feeding_points"]
        
        print("Preprocessing images...")
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        preprocess_images(video_dir, frame_names)

        # Run SAM
        print("Running SAM for segmentation...")
        mask = self.run_sam(
            model_cfg=self.model_cfg,
            sam2_checkpoint=self.sam2_checkpoint,
            device=device,
            video_dir=video_dir,
            point_labels=feeding_points
        )
        
        # Clear annotations outside the specified range
        stopper_points = processed_data["stopper_points"]
        start_index = stopper_points["first_sequence_max"]
        end_index = stopper_points["second_sequence_min"]
        print(f"Clearing annotations outside the range: {start_index} to {end_index}")
        self.clear_outside_range(mask, start_index, end_index)
        
        seg_results = {
            "pts": processed_data,
            "mask": mask
        }
        
        # Create a results directory for the current key
        base_filename = os.path.splitext(key)[0]
        current_results_dir = os.path.join(self.results_dir, base_filename)
        os.makedirs(current_results_dir, exist_ok=True)
        
        # # Get base filename without any extensions using os.path.splitext
        # base_filename = os.path.splitext(key)[0]
        # pkl_path = os.path.join(self.results_dir, f"{base_filename}.pkl")
        # with open(pkl_path, 'wb') as f:
        #     pickle.dump(seg_results, f)
            
        # Convert mask to 3D annotation array
        data_annotation = self.convert_masks_to_annotation(mask, data.shape)
        
        # Save the annotation array as a mrc file
        mrc_path = self.save_annotation_mrc(data_annotation, key, current_results_dir)
        
        # Save the segmentation results and update the results dictionary
        results[key] = {"seg_results": seg_results, 
                        "membrane_annotation_path": mrc_path}
        
        
        print(f"Analysis complete for key: {key}. Results saved.")

    def run_sam(
            self,
            model_cfg,
            sam2_checkpoint,
            device,
            video_dir,
            point_labels,
            ):
        """
        Run the SAM2 model for video segmentation.
        This method initializes the SAM2 model, processes video frames, and performs both forward
        and backward propagation for mask prediction.
        
        Args:
            model_cfg (str): Model configuration path
            sam2_checkpoint (str): Path to model checkpoint
            device (str): Computing device (cuda/cpu/mps)
            video_dir (str): Directory containing video frames
            point_labels (dict): Dictionary containing frame-wise point annotations with structure:
                {frame_idx: {"obj_id": int, "points": array, "labels": array}}
            
        Returns:
            dict: Frame-wise segmentation masks with structure:
                {frame_idx: {obj_id: binary_mask_array}}
        """
        
        # Change to the ABLA root directory
        original_dir = os.getcwd()
        os.chdir("/home/matiasgp/Desktop/ABLA")
        
        try:

            # Clear any existing Hydra initialization
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize Hydra with the correct base path
            initialize(version_base=None, config_path="sam2/sam2")
            
            # Build the predictor
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
            
            inference_state = predictor.init_state(video_path=video_dir)
            predictor.reset_state(inference_state)

            # Iterate over the annotations to add points for each frame
            for frame_idx, data in point_labels.items():
                obj_id = data["obj_id"]
                points = data["points"]
                labels = data["labels"]
                
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            # Run propagation throughout the video and collect the results
            video_segments = {}
            
            # Forward propagation
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Backward propagation
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            return video_segments

        finally:
            # Restore original working directory
            os.chdir(original_dir)

    def clear_outside_range(self, annotations, start=None, end=None):
        """
        Clear annotations outside the specified frame range by setting masks to zero.
        Used to remove predictions in frames that are not part of the region of interest.
        
        Args:
            annotations (dict): Dictionary of frame annotations
            start (int, optional): Starting frame index (inclusive)
            end (int, optional): Ending frame index (inclusive)
        """
        for key in annotations:
            if (start is not None and key < start) or (end is not None and key > end):
                mask_shape = annotations[key][1].shape
                annotations[key][1] = np.zeros(mask_shape, dtype=bool)
        print(f"Annotations cleared outside the range: {start} to {end}")

    def get_tomogram(self, data_path):
        """
        Load tomogram data from an MRC file.
        
        Args:
            data_path (str): Path to the MRC file
            
        Returns:
            numpy.ndarray: Loaded tomogram data as a numpy array
        """
        tomogram = None
        with mrcfile.open(data_path) as mrc:
            tomogram = mrc.data.copy()
        return tomogram
    
    def save_annotation_mrc(self, data_annotation, key, results_dir):
        """
        Save annotation data as an MRC file. Flips the data vertically before saving.
        
        Args:
            data_annotation (numpy.ndarray): The 3D annotation data to save
            key (str): Identifier used to generate the filename
            results_dir (str): Directory to save the results in
        """
        base_key = os.path.splitext(key)[0]
        mrc_path = os.path.join(results_dir, f"{base_key}_annotation.mrc")
        
        try:
            # Flip the data vertically (along axis 1 which is height)
            data_to_save = np.flip(data_annotation, axis=1)
            data_to_save = data_to_save.astype(np.int8)
            
            with mrcfile.new(mrc_path, overwrite=True) as mrc:
                mrc.header.mode = 0  # Mode 0 is for signed int8
                mrc.set_data(data_to_save)
                
            print(f"Successfully saved annotation to: {mrc_path}")
            print(f"Saved data shape: {data_to_save.shape}")
            print(f"Data type: {data_to_save.dtype}")
            
        except Exception as e:
            print(f"Error saving MRC file: {e}")
            try:
                data_to_save = np.flip(data_annotation, axis=1)
                with mrcfile.new(mrc_path, overwrite=True) as mrc:
                    mrc.set_data(data_to_save.astype(np.float32))
                print(f"Successfully saved using float32 format")
            except Exception as e:
                print(f"Alternative saving method also failed: {e}")
                
        return mrc_path

    def convert_masks_to_annotation(self, mask, tomo_shape):
        """
        Convert 2D frame-wise masks to a 3D annotation array aligned with tomogram dimensions.
        Handles different orientations based on the shortest axis of the tomogram.
        
        Args:
            mask (dict): Dictionary of frame masks, where each frame contains object masks
            tomo_shape (tuple): Shape of the tomogram (depth, height, width)
            
        Returns:
            numpy.ndarray: 3D annotation array matching tomogram dimensions
        """
        # Find shortest axis
        shortest_axis = np.argmin(tomo_shape)
        
        # Initialize data_annotation with tomogram dimensions
        data_annotation = np.zeros(tomo_shape, dtype=np.int8)
        
        # Iterate through the frames in the mask dictionary
        for frame_idx, frame_dict in mask.items():
            # Skip if frame index is out of bounds
            if frame_idx >= tomo_shape[shortest_axis]:
                continue
            
            # For each object ID in the frame
            for obj_id, mask_data in frame_dict.items():
                # Remove extra dimension if present
                mask_2d = np.squeeze(mask_data)
                
                # Resize the mask to match the correct dimensions
                if shortest_axis == 0:
                    # Frames are along depth axis
                    resized_mask = cv2.resize(
                        mask_2d.astype(np.uint8),
                        (tomo_shape[2], tomo_shape[1]),  # width, height
                        interpolation=cv2.INTER_NEAREST
                    )
                    data_annotation[frame_idx] = resized_mask
                elif shortest_axis == 1:
                    # Frames are along height axis
                    resized_mask = cv2.resize(
                        mask_2d.astype(np.uint8),
                        (tomo_shape[2], tomo_shape[0]),  # width, depth
                        interpolation=cv2.INTER_NEAREST
                    )
                    data_annotation[:, frame_idx, :] = resized_mask
                else:
                    # Frames are along width axis
                    resized_mask = cv2.resize(
                        mask_2d.astype(np.uint8),
                        (tomo_shape[1], tomo_shape[0]),  # height, depth
                        interpolation=cv2.INTER_NEAREST
                    )
                    data_annotation[:, :, frame_idx] = resized_mask
        
        return data_annotation

# Add a preprocessing step for the images
def preprocess_images(video_dir, frame_names):
    """Preprocessing utility function"""
    images = []
    for frame_name in frame_names:
        img_path = os.path.join(video_dir, frame_name)
        # Read image and convert to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return images


### VISUALIZATION FUNCTIONS ###
def plot_and_save_frames(video_segments, frame_names, video_dir, output_dir, vis_stride=1):
    """Visualization utility function"""
    # ... implementation ...

def show_mask(mask, ax, obj_id=None):
    """Mask display utility function"""
    # ... implementation ...

def create_video(output_dir, fps=30):
    """Video creation utility function"""
    # ... implementation ...
