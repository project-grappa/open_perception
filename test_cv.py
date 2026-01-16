import cv2
import threading
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os

class ImageViewer:
    def __init__(self):
        self.current_image = None
        self.display_mode = 'camera'  # 'camera' or 'image'
        self.running = True
        self.lock = threading.Lock()
        self.drag_drop_window = None
        self.drag_drop_active = False
        
    def load_image(self, filepath):
        """Load an image from file"""
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                with self.lock:
                    self.current_image = img
                    self.display_mode = 'image'
                print(f"Loaded image: {filepath}")
                return True
            else:
                print(f"Error: Could not load image from {filepath}")
        return False
    
    def setup_drag_drop_window(self):
        """Create the drag and drop window"""
        root = TkinterDnD.Tk()
        root.title("Drag & Drop Image Loader")
        root.geometry("400x200")
        
        label = ttk.Label(
            root, 
            text="Drag and drop image files here\n\nSupported formats: .jpg, .jpeg, .png, .bmp, etc.\n\nImages will appear in the OpenCV window\n\nPress 'd' in OpenCV window to close",
            justify=tk.CENTER,
            padding=20
        )
        label.pack(expand=True, fill=tk.BOTH)
        
        def on_drop(event):
            """Handle dropped files"""
            files = root.tk.splitlist(event.data)
            for filepath in files:
                # Remove curly braces if present (Windows paths)
                filepath = filepath.strip('{}')
                self.load_image(filepath)
                # Only load the first valid image
                break
        
        # Register the drop target
        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', on_drop)
        
        # Configure drop zone styling
        label.configure(relief=tk.RIDGE, borderwidth=2)
        
        self.drag_drop_window = root
    
    def camera_thread(self):
        """Thread for camera streaming and image display with integrated drag-drop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Warning: Could not open camera, image viewer mode only")
            cap = None
        
        print("Controls:")
        print("  - Press 'd' to toggle drag & drop window")
        print("  - Press 'c' to switch back to camera")
        print("  - Press 'q' to quit")
        
        while self.running:
            with self.lock:
                current_mode = self.display_mode
                current_image = self.current_image
            
            if current_mode == 'camera' and cap is not None:
                # Capture frame-by-frame from camera
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('Image Viewer (Drag & Drop Images)', frame)
                else:
                    print("Warning: Failed to capture frame")
            elif current_mode == 'image' and current_image is not None:
                # Display the loaded image
                cv2.imshow('Image Viewer (Drag & Drop Images)', current_image)
            
            # Update tkinter drag-drop window if active
            if self.drag_drop_active and self.drag_drop_window:
                try:
                    self.drag_drop_window.update()
                except tk.TclError:
                    # Window was closed
                    self.drag_drop_active = False
                    self.drag_drop_window = None
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('c'):
                # Switch back to camera mode
                with self.lock:
                    self.display_mode = 'camera'
                print("Switched to camera mode")
            elif key == ord('d'):
                # Toggle drag and drop window
                if self.drag_drop_active:
                    # Close the window
                    if self.drag_drop_window:
                        self.drag_drop_window.destroy()
                        self.drag_drop_window = None
                    self.drag_drop_active = False
                    print("Closed drag & drop window")
                else:
                    # Open the window
                    self.setup_drag_drop_window()
                    self.drag_drop_active = True
                    print("Opened drag & drop window")
        
        # Release resources
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if self.drag_drop_window:
            self.drag_drop_window.destroy()

def main():
    viewer = ImageViewer()
    
    # Run camera thread directly (no separate thread needed)
    viewer.camera_thread()

if __name__ == "__main__":
    main()