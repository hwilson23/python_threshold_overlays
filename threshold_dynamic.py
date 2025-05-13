import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QSlider, QScrollArea, QGroupBox, QCheckBox, QComboBox, QDoubleSpinBox,
                            QColorDialog,QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from tifffile import tifffile as tiff


class ThresholdRange:
    def __init__(self, name, min_val=0, max_val=255, color=(255, 0, 0), enabled=True):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.color = color  # RGB tuple
        self.enabled = enabled


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        self.thresholds = []
    
    def plot_histogram(self, img, bins=256):
        self.axes.clear()
        
        if img is not None:
            #hist, bins = np.histogram(~np.isnan(img.flatten()), bins,density=True)
            nanpos = np.isnan(img.flatten())
            #hist, bins,_ = matplotlib.pyplot.hist(img.flatten()[~nanpos],histtype='bar',facecolor='gray')
            hist, bins = np.histogram(img.flatten()[~nanpos],bins)
                        
            #self.axes.bar(bins[:-1], hist, width=bins[1]-bins[0], color='gray', edgecolor = 'black')
            bin_centers = (bins[:-1]+bins[1:])/2
            self.axes.bar(bin_centers, hist, width=bins[1]-bins[0], color='gray', edgecolor = 'black')
            self.axes.set_title('REFERENCE Image Histogram')
            self.axes.set_xlabel('Pixel Value')
            self.axes.set_ylabel('Count')
            
            for t_range in self.thresholds:
                if t_range.enabled:
                    r, g, b = t_range.color
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    self.axes.axvline(x=t_range.min_val, color=color, linestyle='--')
                    self.axes.axvline(x=t_range.max_val, color=color, linestyle='-')
                    self.axes.set_xlim(xmax=t_range.max_val)
        
        self.fig.tight_layout()
        self.draw()
        
    
    def set_thresholds(self, thresholds):
        self.thresholds = thresholds


class ImageThresholdAdjuster(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Threshold Color Overlay")
        self.setGeometry(100, 100, 1400, 900)
                
        # Image and threshold variables
        self.images = []
        self.current_image_index = -1
        self.reference_image_index = -1
        
        self.intensity_min = 0
        self.intensity_max = 255
        
        # Define threshold ranges with different colors
        self.threshold_ranges = [
            ThresholdRange("Range 1", 0, 0, (255, 0, 0)),    # Red
            ThresholdRange("Range 2", 0, 0, (0, 255, 0)),    # Green
            ThresholdRange("Range 3", 0, 0, (0, 0, 255))     # Blue
        ]
        
        self.range_controls = []
        self.use_reference_masks = True  # Always use reference masks
        self.histogram_canvas = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Top controls layout
        top_controls = QHBoxLayout()
        
        # Load image button
        load_button = QPushButton("Load Images")
        load_button.clicked.connect(self.load_images)
        top_controls.addWidget(load_button)
        
        # Reference image selection
        reference_layout = QHBoxLayout()
        reference_label = QLabel("Reference Image:")
        reference_label.setScaledContents(True)
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(self.set_reference_image)
        reference_layout.addWidget(reference_label)
        reference_layout.addWidget(self.reference_combo)
        
        reference_group = QGroupBox("Reference Settings")
        reference_group.setLayout(reference_layout)
        top_controls.addWidget(reference_group)
        
        # Add new threshold range button
        add_range_button = QPushButton("Add Threshold Range")
        add_range_button.clicked.connect(self.add_threshold_range)
        top_controls.addWidget(add_range_button)
        
        # Navigation buttons
        prev_button = QPushButton("Previous Image")
        prev_button.clicked.connect(self.show_previous_image)
        next_button = QPushButton("Next Image")
        next_button.clicked.connect(self.show_next_image)
        top_controls.addWidget(prev_button)
        top_controls.addWidget(next_button)
        
        # Intensity range label
        self.intensity_range_label = QLabel(f"Image Intensity Range: [{self.intensity_min}, {self.intensity_max}]")
        top_controls.addWidget(self.intensity_range_label)
        
        main_layout.addLayout(top_controls)
        
        # Create threshold control groups
        threshold_controls = QVBoxLayout()
        for i, threshold_range in enumerate(self.threshold_ranges):
            group = self.create_threshold_group(i, threshold_range)
            threshold_controls.addWidget(group)
        
        # Add threshold controls to a scrollable area
        threshold_scroll = QScrollArea()
        threshold_scroll.setWidgetResizable(True)
        threshold_container = QWidget()
        threshold_container.setLayout(threshold_controls)
        threshold_scroll.setWidget(threshold_container)
        
        # Image display area
        #self.image_scroll_area1 = QScrollArea()
        #self.image_scroll_area1 = QScrollArea()
        self.image_scroll_area1 = QLabel("Load a .tif or .tiff image to begin")
        self.image_scroll_area2 = QLabel()
        self.image_scroll_area1.setScaledContents(True)
        self.image_scroll_area1.setAlignment(Qt.AlignCenter)
        #self.image_scroll_area2.setWidgetResizable(True)
        self.image_scroll_area2.setScaledContents(True)
        self.image_scroll_area2.setAlignment(Qt.AlignCenter)
        
        self.image_display1 = QLabel("Load a .tif or .tiff image to begin")
        self.image_display1.setAlignment(Qt.AlignCenter)
        self.image_display2 = QLabel()
        self.image_display2.setAlignment(Qt.AlignCenter)
        #self.image_scroll_area1.setWidget(self.image_display1)
        #self.image_scroll_area2.setWidget(self.image_display2)
        
    
        
        # Create histogram canvas
        self.histogram_canvas = HistogramCanvas()
        
        # Create a vertical layout for image and histogram
        image_layout = QVBoxLayout()
        #image_layout.addWidget(self.image_scroll_area1, 3)  # ratio
        image_layout.addWidget(self.image_display1,3)
        #image_layout.addWidget(self.image_scroll_area2,3)
        image_layout.addWidget(self.image_display2,3)
        image_layout.addWidget(self.histogram_canvas, 3)

        
        
        
        # Create image display widget
        image_display_widget = QWidget()
        image_display_widget.setLayout(image_layout)
        
        # Create a horizontal layout for threshold controls and image display
        h_layout = QHBoxLayout()
        h_layout.addWidget(threshold_scroll, 1)  # 1:3 ratio
        h_layout.addWidget(image_display_widget, 3)
        
        main_layout.addLayout(h_layout)
        
        # Set up the layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status bar for information
        self.statusBar().showMessage("Ready")

    

    def create_threshold_group(self, idx, threshold_range):
        def on_slider_change(value):
            slider_scale = int(1/self.float_step)
            float_value = value / slider_scale
            min_spinbox.blockSignals(True)
            min_spinbox.setValue(float_value)
            min_spinbox.blockSignals(False)
            min_label.setText(f"Min: {float_value:.2f}")
            self.update_min_threshold(idx, float_value, min_label)

    # Sync spinbox → slider
        def on_spinbox_change(value):
            slider_scale = int(1/self.float_step)
            slider_value = int(value * slider_scale)
            min_slider.blockSignals(True)
            min_slider.setValue(slider_value)
            min_slider.blockSignals(False)
            min_label.setText(f"Min: {value:.2f}")
            self.update_min_threshold(idx, value, min_label)
        
        
        """Create UI controls for a threshold range"""
        group = QGroupBox(threshold_range.name)
        group_layout = QVBoxLayout()
        
        # Enable/disable checkbox
        enable_checkbox = QCheckBox("Enable")
        enable_checkbox.setChecked(threshold_range.enabled)
        enable_checkbox.stateChanged.connect(lambda state, idx=idx: self.toggle_range(idx, state))
        group_layout.addWidget(enable_checkbox)
        
        # Color picker button
        color_button = QPushButton("Select Color")
        color_button.setStyleSheet(f"background-color: rgb{threshold_range.color}")
        color_button.clicked.connect(lambda _, idx=idx: self.select_color(idx))
        group_layout.addWidget(color_button)
        
        # Min threshold control
        self.float_step = 0.1
        min_layout = QHBoxLayout()
        min_label = QLabel(f"Min: {threshold_range.min_val}")
        #min_slider = QSlider(Qt.Horizontal)
        min_spinbox = QDoubleSpinBox()
        min_spinbox.setRange(self.intensity_min,self.intensity_max)
        min_spinbox.setSingleStep(self.float_step)
        min_spinbox.setDecimals(2)
        #min_slider.setRange(int(self.intensity_min*int(1/self.float_step)), int(self.intensity_max*int(1/self.float_step)))
        #min_slider.setValue(int(threshold_range.min_val*int(1/self.float_step)))
        #min_slider.valueChanged.connect(lambda value, idx=idx, lbl=min_label: self.update_min_threshold(idx, value, lbl))
        #min_slider.valueChanged.connect(on_slider_change)
        #min_spinbox.valueChanged.connect(on_spinbox_change)
        min_spinbox.valueChanged.connect(lambda value, idx=idx, lbl=min_label: self.update_min_threshold(idx, value, lbl))
        min_layout.addWidget(min_label)
        #min_layout.addWidget(min_slider)
        min_layout.addWidget(min_spinbox)
        group_layout.addLayout(min_layout)
        
        
        # Max threshold control
        max_layout = QHBoxLayout()
        max_label = QLabel(f"Max: {threshold_range.max_val}")
        #max_slider = QSlider(Qt.Horizontal)
        max_spinbox = QDoubleSpinBox()
        max_spinbox.setRange(self.intensity_min,self.intensity_max)
        max_spinbox.setSingleStep(self.float_step)
        max_spinbox.setDecimals(2)
        #max_slider.setRange(int(self.intensity_min*int(1/self.float_step)), int(self.intensity_max*int(1/self.float_step)))
        #max_slider.setValue(int(threshold_range.max_val*int(1/self.float_step)))
        #max_slider.valueChanged.connect(lambda value, idx=idx, lbl=max_label: self.update_max_threshold(idx, value, lbl))
        #max_slider.valueChanged.connect(on_slider_change)
        #max_spinbox.valueChanged.connect(on_spinbox_change)
        max_spinbox.valueChanged.connect(lambda value, idx=idx, lbl=max_label: self.update_max_threshold(idx, value, lbl))
        max_layout.addWidget(max_label)
       # max_layout.addWidget(max_slider)
        max_layout.addWidget(max_spinbox)
        group_layout.addLayout(max_layout)
        
        group.setLayout(group_layout)
        
        # Store controls for later access
        self.range_controls.append({
            'enable_checkbox': enable_checkbox,
            'color_button': color_button,
            'min_label': min_label,
            #'min_slider': min_slider,
            'max_label': max_label,
            #'max_slider': max_slider,
            'min_spinbox': min_spinbox,
            'max_spinbox': max_spinbox,
        })
        
        return group
    
    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "", "Image Files (*.tif *.tiff)"
        )
        
        if not file_paths:
            return
        
        self.images = []
        first_image_shape = None
        # Load all images
        for file_path in file_paths:
            try:
                # Load the image with original intensity values
                img = tiff.imread(file_path)
                if img is None:
                    self.statusBar().showMessage(f"Failed to load {file_path}")
                    continue
                
                # Check for consistent shape
                if first_image_shape is None:
                    first_image_shape = img.shape
                elif img.shape != first_image_shape:
                    filename = file_path.split('/')[-1]
                    if '\\' in file_path:
                        filename = file_path.split('\\')[-1]
                    QMessageBox.warning(
                        self,
                        "Image Size Mismatch",
                        f"Image '{filename}' has size {img.shape}, but expected {first_image_shape}. It will not be loaded."
                    )
                    continue  # Skip this mismatched image

                # Find min and max values
                img_min = np.nanmin(img)
                if np.isnan(img_min):
                    img_min = 0
                    print('img_min reset to 0 since min was nan')
                img_max = np.nanmax(img)
                print(f'img_max = {img_max}')
                
                # Get filename for reference dropdown
                filename = file_path.split('/')[-1]
                if '\\' in file_path:  # Handle Windows paths
                    filename = file_path.split('\\')[-1]
                
                # Store the image
                self.images.append({
                    'path': file_path,
                    'filename': filename,
                    'original': img,
                    'processed': None,
                    'min_val': img_min,
                    'max_val': img_max,
                    'masks': []  # Will store masks for each threshold range
                })
            except Exception as e:
                self.statusBar().showMessage(f"Error loading {file_path}: {str(e)}")
        
        if self.images:
            # Update reference image dropdown
            self.reference_combo.clear()
            for i, img_data in enumerate(self.images):
                self.reference_combo.addItem(f"{i+1}: {img_data['filename']}")
                index = self.reference_combo.findText(f"{i+1}: {img_data['filename']}")
                self.reference_combo.setItemData(index,img_data['filename'],role=Qt.ToolTipRole)
            
            # Set current image as reference by default
            self.current_image_index = 0
            self.reference_image_index = 0
            
            # Set intensity range based on reference image
            self.update_intensity_range()
            
            # Generate masks for reference image
            self.generate_reference_masks()
            
            # Process and display current image
            self.process_current_image()
            self.display_current_image()
            # Update histogram
            self.update_histogram()
    
    def update_intensity_range(self):
        """Update intensity range based on reference image"""
        if self.reference_image_index < 0 or self.reference_image_index >= len(self.images):
            return
            
        ref_img = self.images[self.reference_image_index]
        self.intensity_min = int(ref_img['min_val'])
        self.intensity_max = int(ref_img['max_val'])
        self.intensity_range_label.setText(f"Reference Image Intensity Range: [{self.intensity_min}, {self.intensity_max}]")
        
        # Update all sliders to use the new intensity range
        for controls in self.range_controls:
            #controls['min_slider'].setRange(self.intensity_min, self.intensity_max)
            #controls['max_slider'].setRange(self.intensity_min, self.intensity_max)
            controls['min_spinbox'].setRange(self.intensity_min, self.intensity_max)
            controls['max_spinbox'].setRange(self.intensity_min, self.intensity_max)
            
        
        # Initialize threshold ranges based on the intensity range
        range_size = (self.intensity_max - self.intensity_min) / len(self.threshold_ranges)
        
        for i, threshold_range in enumerate(self.threshold_ranges):
            min_val = int(self.intensity_min + i * range_size)
            if np.isnan(min_val):
                min_val = 0
                print('min val was nan, reset to 0')
            max_val = int(min_val + range_size)
            
            # Update the threshold range
            threshold_range.min_val = min_val
            threshold_range.max_val = max_val
            
            # Update the UI controls
            #self.range_controls[i]['min_slider'].setValue(int(min_val*int(1/self.float_step)))
            #self.range_controls[i]['max_slider'].setValue(int(max_val*int(1/self.float_step)))
            self.range_controls[i]['min_spinbox'].setValue(min_val)
            self.range_controls[i]['max_spinbox'].setValue(max_val)
            self.range_controls[i]['min_label'].setText(f"Min: {min_val}")
            self.range_controls[i]['max_label'].setText(f"Max: {max_val}")
    
    def select_color(self, range_idx):
        r, g, b = self.threshold_ranges[range_idx].color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(initial_color, self, f"Select Color for {self.threshold_ranges[range_idx].name}")
        
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.threshold_ranges[range_idx].color = new_color
            self.range_controls[range_idx]['color_button'].setStyleSheet(f"background-color: rgb{new_color}")
            
            # Update reference masks with new color
            self.generate_reference_masks()
            
            # Update display
            self.process_current_image()
            self.display_current_image()
            self.update_histogram()
    
    def toggle_range(self, range_idx, state):
        self.threshold_ranges[range_idx].enabled = (state == Qt.Checked)
        
        # Update reference masks with new enabled state
        self.generate_reference_masks()
        
        # Update display
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()
    
    def update_min_threshold(self, range_idx, value, label):
        self.threshold_ranges[range_idx].min_val = value
        label.setText(f"Min: {value}")
        
        # Ensure min <= max
        if value > self.threshold_ranges[range_idx].max_val:
            #self.range_controls[range_idx]['max_slider'].blockSignals(True)
            #self.range_controls[range_idx]['max_slider'].setValue(int(value*int(1/self.float_step)))
            self.range_controls[range_idx]['max_spinbox'].setValue(value)
            #self.range_controls[range_idx]['max_slider'].blockSignals(False)
        
        # Update reference masks with new threshold
        self.generate_reference_masks()
        
        # Update display
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()
        
    def update_max_threshold(self, range_idx, value, label):
        self.threshold_ranges[range_idx].max_val = value
        label.setText(f"Max: {value}")
        
        # Ensure max >= min
        if value < self.threshold_ranges[range_idx].min_val:
            #self.range_controls[range_idx]['min_slider'].blockSignals(True)
            #self.range_controls[range_idx]['min_slider'].setValue(int(value*int(1/self.float_step)))
            self.range_controls[range_idx]['min_spinbox'].setValue(value)
            #self.range_controls[range_idx]['min_slider'].blockSignals(False)
            

        # Update reference masks with new threshold
        self.generate_reference_masks()
        
        # Update display
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()
    
    def add_threshold_range(self):
        """Add a new threshold range with default values"""
        # Calculate a new color - try to make it distinct from existing ones
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        
        # Use a color not already in use, or default to white
        used_colors = [r.color for r in self.threshold_ranges]
        new_color = next((c for c in colors if c not in used_colors), (255, 255, 255))
        
        # Create new range with values in the current intensity range
        new_idx = len(self.threshold_ranges)
        mid_point = (self.intensity_min + self.intensity_max) // 2
        new_range = ThresholdRange(f"Range {new_idx + 1}", 
                                  mid_point - (self.intensity_max - self.intensity_min) // 6, 
                                  mid_point + (self.intensity_max - self.intensity_min) // 6, 
                                  new_color)
        self.threshold_ranges.append(new_range)
        
        # Create UI controls for new range
        group = self.create_threshold_group(new_idx, new_range)
        
        # Add to layout
        #threshold_container = self.findChild(QScrollArea).widget()
        threshold_container = self.findChild(QLabel).widget()
        threshold_container.layout().addWidget(group)
        
        # Update reference masks with new range
        self.generate_reference_masks()
        
        # Update display
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()
    
    def set_reference_image(self, index):
        """Set the reference image to use for thresholding"""
        if index >= 0 and index < len(self.images):
            self.reference_image_index = index
            
            # Update intensity range based on reference image
            self.update_intensity_range()
            
            # Generate masks for the reference image
            self.generate_reference_masks()
            
            # Update histogram with reference image
            self.update_histogram()
            
            # Process and display current image
            self.process_current_image()
            self.display_current_image()
            self.update_histogram()
            
            self.statusBar().showMessage(f"Using reference image: {self.images[self.reference_image_index]['filename']}")
    
    def generate_reference_masks(self):
        """Generate masks for the reference image based on current thresholds"""
        if self.reference_image_index < 0 or self.reference_image_index >= len(self.images):
            return
        
        ref_img = self.images[self.reference_image_index]
        ref_data = ref_img['original']
        
        # Clear existing masks
        ref_img['masks'] = []
        
        # Generate a mask for each threshold range
        for threshold_range in self.threshold_ranges:
            if threshold_range.enabled:
                mask = (ref_data >= threshold_range.min_val) & (ref_data <= threshold_range.max_val)
                mask = mask.astype(np.uint8)
            else:
                mask = np.zeros(ref_data.shape, dtype=np.uint8)
            
            ref_img['masks'].append({
                'mask': mask,
                'color': threshold_range.color,
                'enabled': threshold_range.enabled
            })
    
    def update_histogram(self):
        """Update the histogram display with current image data"""
        if self.reference_image_index >= 0 and self.reference_image_index < len(self.images):
            ref_img = self.images[self.reference_image_index]
            # Set thresholds on the histogram for visualization
            self.histogram_canvas.set_thresholds(self.threshold_ranges)
            # Plot the histogram
            self.histogram_canvas.plot_histogram(ref_img['original'])
    
    def process_image(self, img_data):
        """Process the image by applying colored overlays based on reference masks"""
        # Get the original image
        img = img_data['original']
        
        # Create an RGB overlay image for the colored thresholds
        h, w = img.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create a mask to keep track of overlapping regions for proper blending
        mask_applied = np.zeros((h, w), dtype=np.uint8)
        
        # Use masks from reference image
        if self.reference_image_index >= 0:
            ref_img = self.images[self.reference_image_index]
            
            # Apply each mask with its color
            for mask_data in ref_img['masks']:
                if not mask_data['enabled']:
                    continue
                
                # Only apply color to pixels that haven't been colored yet
                unused_pixels = ~mask_applied.astype(bool)
                new_mask = mask_data['mask'].astype(bool) & unused_pixels
                
                # Apply color to the overlay
                overlay[new_mask] = mask_data['color']
                
                # Update mask of applied pixels
                mask_applied[new_mask] = 1
        
        # Create a normalized version of the original for display
        norm_img = img.copy()
        if norm_img.dtype != np.uint8:
            # Scale to 0-255 for display purposes only
            img_min = np.min(norm_img)
            if np.isnan(img_min):
                img_min = 0
                print('img_min reset to 0 since min was nan')
            img_max = np.max(norm_img)
            if img_max > img_min:
                norm_img = ((norm_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                norm_img = np.zeros_like(norm_img, dtype=np.uint8)
        
        # Convert the normalized grayscale to RGB for display
        img_rgb = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        
        # Blend overlay with RGB image (opacity)
        alpha = 0.3
        blended = np.zeros_like(img_rgb)
        cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0, blended)
        
        return blended
    
    def process_current_image(self):
        if not self.images or self.current_image_index < 0:
            return
        
        current = self.images[self.current_image_index]
        current['processed'] = self.process_image(current)
    
    def display_current_image(self):
        if not self.images or self.current_image_index < 0:
            return
        
        current = self.images[self.current_image_index]
        
        if current['processed'] is None:
            current['processed'] = self.process_image(current)
        
        # Convert processed image to QImage for display
        h, w, c = current['processed'].shape
        bytes_per_line = 3 * w
        q_img = QImage(current['processed'].data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap1 = QPixmap.fromImage(q_img)
        if w or h > 2000:
            pixmap1scaled = self.pixmap1.scaled(512,512,Qt.KeepAspectRatio, Qt.SmoothTransformation)
            

        #make 16 bit grayscale
        grayim = np.array(current['original'].data, dtype=np.uint8)
        q_img2 = QImage(grayim,w,h,w, QImage.Format_Grayscale8)
        self.pixmap2 = QPixmap.fromImage(q_img2)
        if w or h > 2000:
            pixmap2scaled = self.pixmap2.scaled(512,512,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Display image
        self.image_display1.setPixmap(pixmap1scaled)
        self.image_display2.setPixmap(pixmap2scaled)
        
        self.image_display1.setAlignment(Qt.AlignCenter)
        self.image_display2.setAlignment(Qt.AlignCenter)
        
        # Update status bar
        img_info = f"Image: {current['filename']} ({w}x{h}), total of {len(self.images)} image(s) loaded."
        self.statusBar().showMessage(img_info)
    
    def show_next_image(self):
        if not self.images:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()
    
    def show_previous_image(self):
        if not self.images:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.process_current_image()
        self.display_current_image()
        self.update_histogram()

     

if __name__ == "__main__":
    
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1)
    sys._excepthook = sys.excepthook 
    sys.excepthook = exception_hook 

    app = QApplication(sys.argv)
    window = ImageThresholdAdjuster()
    window.show()
    sys.exit(app.exec_())


