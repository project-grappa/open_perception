import os
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinterdnd2 import DND_FILES, TkinterDnD
from typing import Callable, Optional


class DragDropImageLoader:
    """
    A standalone drag-and-drop image loader that integrates with cv2 event loops.
    Creates a tkinter window that accepts image files via drag-and-drop.
    """
    
    def __init__(self, on_image_loaded: Optional[Callable[[str, any], None]] = None):
        """
        Initialize the drag-and-drop image loader.
        
        Args:
            on_image_loaded: Callback function called when an image is loaded.
                           Signature: on_image_loaded(filepath: str, image: np.ndarray)
        """
        self.on_image_loaded = on_image_loaded
        self.window = None
        self.active = False
        
    def open(self, title: str = "Drag & Drop Image Loader", geometry: str = "400x200"):
        """
        Open the drag-and-drop window.
        
        Args:
            title: Window title
            geometry: Window size (format: "widthxheight")
        """
        if self.active:
            return
            
        self.window = TkinterDnD.Tk()
        self.window.title(title)
        self.window.geometry(geometry)
        
        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Drop zone label
        label = ttk.Label(
            main_frame, 
            text="Drag and drop image or video files here\n\n"
                 "Supported formats:\n"
                 "Images: .jpg, .jpeg, .png, .bmp, etc.\n"
                 "Videos: .mp4, .avi, .mov, .mkv, etc.\n\n"
                 "Files will be loaded into the viewer",
            justify=tk.CENTER,
            padding=20
        )
        label.pack(expand=True, fill=tk.BOTH)
        
        # Browse button
        browse_button = ttk.Button(
            main_frame,
            text="Browse File",
            command=self._browse_file
        )
        browse_button.pack(pady=(10, 0))
        
        def on_drop(event):
            """Handle dropped files"""
            files = self.window.tk.splitlist(event.data)
            for filepath in files:
                # Remove curly braces if present (Windows paths)
                filepath = filepath.strip('{}')
                self._load_image(filepath)
                # Only load the first valid image
                break
        
        # Register the drop target
        self.window.drop_target_register(DND_FILES)
        self.window.dnd_bind('<<Drop>>', on_drop)
        
        # Configure drop zone styling
        label.configure(relief=tk.RIDGE, borderwidth=2)
        
        self.active = True
        
    def close(self):
        """Close the drag-and-drop window."""
        if self.window:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            self.window = None
        self.active = False
    
    def update(self):
        """
        Update the tkinter window. Should be called regularly from the main event loop.
        Returns False if the window was closed.
        """
        if self.active and self.window:
            try:
                self.window.update()
                return True
            except tk.TclError:
                # Window was closed
                self.active = False
                self.window = None
                return False
        return False
    
    def toggle(self):
        """Toggle the drag-and-drop window open/closed."""
        if self.active:
            self.close()
        else:
            self.open()
    
    def _browse_file(self):
        """Open file dialog to browse for an image or video file."""
        filepath = askopenfilename(
            title="Select an image or video file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self._load_image(filepath)
    
    def _load_image(self, filepath: str):
        """
        Load an image or video from file and call the callback.
        
        Args:
            filepath: Path to the image or video file
        """
        if os.path.isfile(filepath):
            # Try loading as image first
            img = cv2.imread(filepath)
            if img is not None:
                print(f"Loaded image: {filepath}")
                if self.on_image_loaded:
                    self.on_image_loaded(filepath, img)
                return True
            else:
                # If not an image, try as video
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    print(f"Loaded video: {filepath}")
                    cap.release()
                    if self.on_image_loaded:
                        self.on_image_loaded(filepath, None)  # Pass None for videos
                    return True
                else:
                    print(f"Error: Could not load file from {filepath}")
        return False
