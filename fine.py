import modal
from pathlib import Path

# Define Modal image with system-level and pip dependencies
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "libgl1-mesa-glx",  # Required for OpenCV
    "libglib2.0-0",     # Additional system libraries
).pip_install(
    "ultralytics~=8.2.68", 
    "opencv-python-headless~=4.10.0",  # Use headless version for environments without display
    "numpy"
)

# Create persistent volume for models and data
volume = modal.Volume.from_name("yolo-inference")
volume_path = Path("/root/model_data")

# Create Modal app
app = modal.App("yolo-webcam-inference", image=image, volumes={volume_path: volume})

@app.cls(gpu="A10G")
class YOLOWebcamInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(str(volume_path / self.model_path))
        print(f"Loaded model from {self.model_path}")

    @modal.method()
    def live_inference(self, conf_threshold: float = 0.25, output_file: str = "webcam_output.mp4"):
        """
        Perform real-time object detection using video input
        
        Args:
            conf_threshold (float): Confidence threshold for detection
            output_file (str): Name of the output video file
        
        Returns:
            str: Path to the saved output video
        """
        import cv2
        import numpy as np

        # Open video capture (can be webcam or video file)
        cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # If unable to get fps, use a default
        if fps == 0:
            fps = 30
        
        # Create video writer
        output_path = str(volume_path / "webcam_results" / output_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        try:
            frame_count = 0
            max_frames = 500  # Limit recording to prevent extremely large files
            
            while cap.isOpened() and frame_count < max_frames:
                success, frame = cap.read()
                if not success:
                    break
                
                # Perform object detection with tracking
                results = self.model.track(
                    frame, 
                    persist=True,
                    conf=conf_threshold,
                    verbose=False
                )
                
                # Write annotated frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                frame_count += 1
        
        finally:
            # Cleanup
            cap.release()
            out.release()
        
        return output_path

@app.local_entrypoint()
def main(
    model_path: str = "runs/bees-tbdsg/weights/best.pt",
    conf_threshold: float = 0.3,
    output_file: str = "webcam_detection.mp4"
):
    # Create necessary directories in the volume
    volume.mkdir("/root/model_data/webcam_results", parents=True, exist_ok=True)
    
    # Run webcam inference
    result_path = YOLOWebcamInference(model_path).live_inference.remote(
        conf_threshold=conf_threshold,
        output_file=output_file
    )
    
    print(f"\nVideo recording saved to: {result_path}")
    print(f"View results in Modal volume at: {volume_path / 'webcam_results'}")

if __name__ == "__main__":
    main()
