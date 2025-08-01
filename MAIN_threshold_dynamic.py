import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot
matplotlib.use('Qt5Agg')
import os
import textwrap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
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
            pct = np.nanpercentile(img.flatten()[~nanpos],99)
            nonnanimg = img.flatten()[~nanpos]
            hist, bins = np.histogram(nonnanimg[nonnanimg<=pct],bins)
                        
            #self.axes.bar(bins[:-1], hist, width=bins[1]-bins[0], color='gray', edgecolor = 'black')
            bin_centers = (bins[:-1]+bins[1:])/2
            self.axes.bar(bin_centers, hist, width=bins[1]-bins[0], color='gray', edgecolor = 'black')
            self.axes.set_title('REFERENCE Image Histogram - display to 99th percentile')
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

class MatplotlibImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar for zoom, pan, etc.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        
    def update_images(self, rgb_image, grayscale_image, filename, w, h):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Display RGB image (left subplot - image_display1 equivalent)
        self.ax1.imshow(rgb_image)

        wrapped_filename_rgb = "\n".join(textwrap.wrap(f'Mask: {filename}',width=40))
        self.ax1.set_title(wrapped_filename_rgb)
        self.ax1.axis('off')
        
        # Display grayscale image (right subplot - image_display2 equivalent)
        self.ax2.imshow(grayscale_image, cmap='gray')
        wrapped_filename_gray = "\n".join(textwrap.wrap(f'Grayscale: {filename}',width=40))
        self.ax2.set_title(f'{wrapped_filename_gray} ({w}x{h})')
        self.ax2.axis('off')
        
        # Adjust layout and refresh
        self.figure.tight_layout()
        self.canvas.draw()
    
    

class ImageThresholdAdjuster(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Threshold Color Overlay")
        self.setGeometry(100, 100, 300, 900)
                
        # Image and threshold variables
        self.images = []
        self.current_image_index = -1
        self.reference_image_index = -1
        
        self.intensity_min = 0
        self.intensity_max = 255
        
        # Define threshold ranges with different colors
        self.threshold_ranges = [
            ThresholdRange("Range 1", 0, 0, (107, 255, 15)),    
            ThresholdRange("Range 2", 0, 0, (41, 173, 136)),    
            ThresholdRange("Range 3", 0, 0, (0, 118, 0))
                 
        ]
        
        self.range_controls = []
        self.use_reference_masks = True  # Always use reference masks
        self.histogram_canvas = None
# Enable zoom and pan
        self.figure = Figure(figsize=(10,5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Setup UI
        self.setup_ui()
        
    def display_current_image(self):
        if not self.images or self.current_image_index < 0:
            return
        
        current = self.images[self.current_image_index]
    
        # Update display range spinboxes with current image values
        display_min = current.get('display_min', np.nanmin(current['original']))
        display_max = current.get('display_max', np.nanmax(current['original']))
    
        self.min_range_spinbox.blockSignals(True)
        self.max_range_spinbox.blockSignals(True)
        self.min_range_spinbox.setValue(display_min)
        self.max_range_spinbox.setValue(display_max)
        self.min_range_spinbox.blockSignals(False)
        self.max_range_spinbox.blockSignals(False)

        # Process image if needed
        if current['processed'] is None:
            current['processed'] = self.process_image_with_custom_range(current)

        # Get image dimensions
        h, w, c = current['processed'].shape
        
        # Prepare grayscale data
        if current['original_mod'] is None:
            graydata_scaled = np.array(current['original'].data)
        else:
            graydata_scaled = np.array(current['original_mod'].data)
        
        # Normalize grayscale data
        graydata_scaled = graydata_scaled * (255 / graydata_scaled.max())
        grayim = np.array(graydata_scaled, dtype=np.uint8)
        
        # Update matplotlib widget instead of Qt labels
        if not hasattr(self, 'image_plot'):
            # Create matplotlib widget if it doesn't exist
            self.image_plot = MatplotlibImage()  # Use the version with toolbar
            
        # Update the matplotlib display
        self.image_plot.update_images(
            current['processed'],  # RGB image for image_display1
            grayim.reshape(h, w),  # Grayscale image for image_display2
            current['filename'],
            w, h
            )
    def update_images(self, rgb_image, grayscale_image, filename, w, h):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Display RGB image (left subplot - image_display1 equivalent)
        self.ax1.imshow(rgb_image)
        self.ax1.set_title(f'RGB: {filename}')
        self.ax1.axis('off')
        
        # Display grayscale image (right subplot - image_display2 equivalent)
        self.ax2.imshow(grayscale_image, cmap='gray')
        self.ax2.set_title(f'Grayscale: {filename} ({w}x{h})')
        self.ax2.axis('off')
        
        # Adjust layout and refresh
        self.figure.tight_layout()
        self.canvas.draw()

    def on_scroll(self, event):
        # This enables zoom with mouse wheel
        # matplotlib already handles this automatically with toolbar
        pass
   
    # Update status bar
        img_info = f"Image: {current['filename']} ({w}x{h}), total of {len(self.images)} image(s) loaded."
        self.statusBar().showMessage(img_info)

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # --- Top Controls ---
        top_controls = QHBoxLayout()

        load_button = QPushButton("Load Images")
        load_button.clicked.connect(self.load_images)
        top_controls.addWidget(load_button)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_images)
        top_controls.addWidget(save_button)

        reference_layout = QHBoxLayout()
        reference_label = QLabel("Reference Image:")
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(self.set_reference_image)
        reference_layout.addWidget(reference_label)
        reference_layout.addWidget(self.reference_combo)

        reference_group = QGroupBox("Reference Settings")
        reference_group.setLayout(reference_layout)
        top_controls.addWidget(reference_group)

        add_range_button = QPushButton("Add Threshold Range")
        add_range_button.clicked.connect(self.add_threshold_range)
        top_controls.addWidget(add_range_button)

        prev_button = QPushButton("Previous Image")
        prev_button.clicked.connect(self.show_previous_image)
        next_button = QPushButton("Next Image")
        next_button.clicked.connect(self.show_next_image)
        top_controls.addWidget(prev_button)
        top_controls.addWidget(next_button)

        self.intensity_range_label = QLabel(f"Image Intensity Range: [{self.intensity_min}, {self.intensity_max}]")
        top_controls.addWidget(self.intensity_range_label)

        # Display range controls (add before layout is finalized!)
        display_range_group = QGroupBox("Display Range")
        display_range_layout = QHBoxLayout()

        min_range_label = QLabel("Min:")
        self.min_range_spinbox = QDoubleSpinBox()
        self.min_range_spinbox.setRange(-1000000, 1000000)
        self.min_range_spinbox.setDecimals(2)
        self.min_range_spinbox.setKeyboardTracking(False)
        self.min_range_spinbox.valueChanged.connect(self.on_display_range_changed)

        max_range_label = QLabel("Max:")
        self.max_range_spinbox = QDoubleSpinBox()
        self.max_range_spinbox.setRange(-1000000, 1000000)
        self.max_range_spinbox.setDecimals(2)
        self.max_range_spinbox.setKeyboardTracking(False)
        self.max_range_spinbox.valueChanged.connect(self.on_display_range_changed)

        reset_range_button = QPushButton("Reset Values")
        reset_range_button.clicked.connect(self.reset_display_range)

        display_range_layout.addWidget(min_range_label)
        display_range_layout.addWidget(self.min_range_spinbox)
        display_range_layout.addWidget(max_range_label)
        display_range_layout.addWidget(self.max_range_spinbox)
        display_range_layout.addWidget(reset_range_button)

        display_range_group.setLayout(display_range_layout)
        top_controls.addWidget(display_range_group)

        main_layout.addLayout(top_controls)

        # --- Threshold Controls Row ---
        threshold_control_layout = QHBoxLayout()
        for i, threshold_range in enumerate(self.threshold_ranges):
            group = self.create_threshold_group(i, threshold_range)
            threshold_control_layout.addWidget(group)

        threshold_scroll = QScrollArea()
        threshold_scroll.setWidgetResizable(True)
        threshold_container = QWidget()
        threshold_container.setLayout(threshold_control_layout)
        threshold_scroll.setWidget(threshold_container)

        main_layout.addWidget(threshold_scroll, stretch=0)

        # --- Image and Histogram Display ---
        self.image_plot = MatplotlibImageWidget()
        self.histogram_canvas = HistogramCanvas()

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_plot, stretch=3)
        image_layout.addWidget(self.histogram_canvas, stretch=1)

        image_display_widget = QWidget()
        image_display_widget.setLayout(image_layout)
        main_layout.addWidget(image_display_widget, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready")

    
    def on_display_range_changed(self):
        """Called when user changes min/max display range values"""
        min_val = self.min_range_spinbox.value()
        max_val = self.max_range_spinbox.value()
        
        # Ensure min <= max
        if min_val > max_val:
            if self.sender() is self.min_range_spinbox:
                self.max_range_spinbox.setValue(min_val)
                max_val = min_val
            else:
                self.min_range_spinbox.setValue(max_val)
                min_val = max_val
        
        self.set_display_range(min_val, max_val)

    def reset_display_range(self):
        """Reset display range to actual min/max values of current image"""
        if not self.images or self.current_image_index < 0:
            return
        
        img = self.images[self.current_image_index]['original']
        img_min = np.nanmin(img)
        if np.isnan(img_min):
            img_min = 0
        img_max = np.nanmax(img)
        
        # Update spinboxes (will trigger on_display_range_changed)
        self.min_range_spinbox.blockSignals(True)
        self.max_range_spinbox.blockSignals(True)
        self.min_range_spinbox.setValue(img_min)
        self.max_range_spinbox.setValue(img_max)
        self.min_range_spinbox.blockSignals(False)
        self.max_range_spinbox.blockSignals(False)
    
        # Apply the range
        self.set_display_range(img_min, img_max)

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
        group_layout = QHBoxLayout()
        
        # Enable/disable checkbox
        enable_checkbox = QCheckBox("Enable")
        enable_checkbox.setChecked(threshold_range.enabled)
        enable_checkbox.stateChanged.connect(lambda state, idx=idx: self.toggle_range(idx, state))
        group_layout.addWidget(enable_checkbox)
        '''
        #apply button
        apply_button = QPushButton("Apply Thresholds")
        apply_button.clicked.connect(lambda _, idx=idx: self.apply_threshold_changes(idx))
        group_layout.addWidget(apply_button)
        '''        
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
        min_spinbox.setKeyboardTracking(False)
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
        max_spinbox.setKeyboardTracking(False)
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
    
    def save_images(self):
        file_name, _ = QFileDialog.getSaveFileName(self,'Save File', '','All Files (*)')
        if file_name:
            current = self.images[self.current_image_index]
            tiff.imwrite(file_name, self.process_image_with_custom_range(current))
            print(f'Saved RGB Image to: {file_name}')
            ref_img = self.images[self.reference_image_index]
            i=1
            for mask_data in ref_img['masks']:
                if not mask_data['enabled']:
                    continue
                ht = os.path.split(file_name)
                tiff.imwrite(f'{ht[0]}//mask{i}_{ht[1]}', mask_data['mask'])
                
                print(f'Saved mask image to: {ht[0]}/mask{i}_{ht[1]}')
                i = i+1

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
                img = np.nan_to_num(img, posinf = float('NaN'))
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
                    'original_mod':None,
                    'processed': None,
                    'min_val': img_min,
                    'max_val': img_max,
                    'masks': []  # Will store masks for each threshold range
                })
            except Exception as e:
                self.statusBar().showMessage(f"Error loading {file_path}: {str(e)}")
                pass
        
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
    '''
    def apply_threshold_changes(self, idx):
        controls = self.range_controls[idx]
        
        min_val = controls['min_spinbox'].value()
        max_val = controls['max_spinbox'].value()

        # Ensure ordering
        if min_val > max_val:
            max_val = min_val
            controls['max_spinbox'].setValue(max_val)

        controls['min_label'].setText(f"Min: {min_val:.2f}")
        controls['max_label'].setText(f"Max: {max_val:.2f}")
        
        self.update_min_threshold(idx, min_val, controls['min_label'])
        self.update_max_threshold(idx, max_val, controls['max_label'])
    '''

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
        threshold_container = self.findChild(QScrollArea).widget()
        #threshold_container = self.findChild(QLabel).widget()
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
    def set_display_range(self, display_min=None, display_max=None):
        """
        Adjust the contrast of the current image by setting custom min/max display values.
        
        Args:
            display_min (float): Minimum intensity value for display range mapping
            display_max (float): Maximum intensity value for display range mapping
        """
        if not self.images or self.current_image_index < 0:
            return
        
        current = self.images[self.current_image_index]
        img = current['original'].copy()
        
        # If no values provided, use the actual min/max
        if display_min is None:
            display_min = np.nanmin(img)
            if np.isnan(display_min):
                display_min = 0
        
        if display_max is None:
            display_max = np.nanmax(img)
        
        # Store the display range values
        current['display_min'] = display_min
        current['display_max'] = display_max
        
        #clip original
        current['original_mod']= np.clip(img, display_min, display_max)
        current['original_mod'] = ((current['original_mod'] - display_min) / (display_max - display_min) * 255).astype(np.uint8)
        # Re-process the images with new display range
        current['processed'] = self.process_image_with_custom_range(current)
        
        # Update the display
        self.display_current_image()
        
        # Update status bar with range info
        current_filename = current['filename']
        img_h, img_w = img.shape
        range_info = f"Image: {current_filename} ({img_w}x{img_h}), Display Range: [{display_min:.2f}, {display_max:.2f}]"
        self.statusBar().showMessage(range_info)

    def process_image_with_custom_range(self, img_data):
        """Process the image with custom min/max display range and apply overlays"""
        # Get original image
        img = img_data['original'].copy()
        
        # Get display range (use stored values or actual min/max)
        display_min = img_data.get('display_min', np.nanmin(img))
        display_max = img_data.get('display_max', np.nanmax(img))
        
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
        
        # Create a normalized version using custom display range
        norm_img = np.clip(img, display_min, display_max)
        norm_img = ((norm_img - display_min) / (display_max - display_min) * 255).astype(np.uint8)
        
        # Convert the normalized grayscale to RGB for display
        img_rgb = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        
        #TODO - better blending?  Blend overlay with RGB image (opacity)
        alpha = 0.2
        blended = np.zeros_like(img_rgb)
        cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0, blended)
        
        
        #return blended
        return overlay
    

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
        
        #return blended
        return overlay
    
    def process_current_image(self):
        if not self.images or self.current_image_index < 0:
            return
        
        current = self.images[self.current_image_index]
        current['processed'] = self.process_image_with_custom_range(current)    

    def ignore_display_current_image(self):
        if not self.images or self.current_image_index < 0:
            return
        '''
        current = self.images[self.current_image_index]
        
        if current['processed'] is None:
            current['processed'] = self.process_image(current)
        '''

        current = self.images[self.current_image_index]
        
        # Update display range spinboxes with current image values
        display_min = current.get('display_min', np.nanmin(current['original']))
        display_max = current.get('display_max', np.nanmax(current['original']))
        
        self.min_range_spinbox.blockSignals(True)
        self.max_range_spinbox.blockSignals(True)
        self.min_range_spinbox.setValue(display_min)
        self.max_range_spinbox.setValue(display_max)
        self.min_range_spinbox.blockSignals(False)
        self.max_range_spinbox.blockSignals(False)
    
        # Process image if needed
        if current['processed'] is None:
            current['processed'] = self.process_image_with_custom_range(current)
    
        
        # Convert processed image to QImage for display
        h, w, c = current['processed'].shape
        '''
        bytes_per_line = 3 * w
        #rgb888 is 24 bit
        q_img = QImage(current['processed'].data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap1 = QPixmap.fromImage(q_img)
        if w > 2000 or h > 2000:
            pixmap1scaled = self.pixmap1.scaled(512,512,Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display1.setPixmap(pixmap1scaled)
        else:
            self.image_display1.setPixmap(self.pixmap1)
        '''
        if current['original_mod'] is None:
            graydata_scaled = np.array(current['original'].data)
        else:
            graydata_scaled = np.array(current['original_mod'].data)

        #TODO make 16/32 bit grayscale/rgb?
        graydata_scaled = graydata_scaled*(255/graydata_scaled.max())
        grayim = np.array(graydata_scaled, dtype=np.uint8)
        '''
        q_img2 = QImage(grayim,w,h,w, QImage.Format_Grayscale8)
        self.pixmap2 = QPixmap.fromImage(q_img2)
        if w >2000 or h > 2000:
            pixmap2scaled = self.pixmap2.scaled(512,512,Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display2.setPixmap(pixmap2scaled)
        else:
            self.image_display2.setPixmap(self.pixmap2)
        
        self.image_display1.setAlignment(Qt.AlignCenter)
        self.image_display2.setAlignment(Qt.AlignCenter)
        '''
         # Update matplotlib widget instead of Qt labels
        if not hasattr(self, 'matplotlib_widget'):
            # Create matplotlib widget if it doesn't exist
            #self.matplotlib_widget = MatplotlibImageWidget()
            # You'll need to add this to your layout where the old image displays were
            # For example: self.main_layout.addWidget(self.matplotlib_widget)
            self.image_plot.addWidget(self.matplotlib_widget)
            #self.image_display2.addWidget(self.matplotlib_widget1)
        # Update the matplotlib display
        self.matplotlib_widget.update_images(
            current['processed'], 
            grayim.reshape(h, w),  # Reshape for grayscale display
            current['filename'],
            w, h
        )
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


