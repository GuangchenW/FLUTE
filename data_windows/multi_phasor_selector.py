#imports
from importlib.resources import files, as_file
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
import os

from . import MultiPhasorGraph
from config import UI_DIR

class MultiPhasorSelector(QtWidgets.QMainWindow):
	"""Dialog for selecting which phasor clouds to display in the multi-view and their colors"""
	def __init__(self, image_arr):
		super(MultiPhasorSelector, self).__init__()
		
		# Load the ui component
		ui_path = UI_DIR / "MultiPhasorSelector.ui"
		uic.loadUi(str(ui_path), self)
		
		self.image_arr = image_arr
		self.selected_images = []
		self.image_colors = {}	# Changed from image_colormaps
		self.image_transparency = {}
		
		# Connect signals
		self.btnSelectAll.clicked.connect(self.select_all)
		self.btnDeselectAll.clicked.connect(self.deselect_all)
		self.btnDisplay.clicked.connect(self.display_selected)
		self.btnClose.clicked.connect(self.close)
		
		# Store available colors for dropdown
		self.available_colors = [
			'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 
			'orange', 'purple', 'lime', 'pink', 'brown', 'navy',
			'teal', 'olive', 'maroon', 'coral', 'indigo', 'turquoise'
		]
		
		# Set up table
		self.populate_table()
		
		self.multi_phasor_window = None
		
	def populate_table(self):
		"""Populates the table with loaded images"""
		self.tableWidget.setRowCount(len(self.image_arr))
		
		for row, image in enumerate(self.image_arr):
			# Image name
			name_item = QtWidgets.QTableWidgetItem(image.name)
			name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)	# Make read-only
			self.tableWidget.setItem(row, 0, name_item)
			
			# Checkbox for inclusion
			checkbox = QtWidgets.QCheckBox()
			cell_widget = QtWidgets.QWidget()
			layout = QtWidgets.QHBoxLayout(cell_widget)
			layout.addWidget(checkbox)
			layout.setAlignment(Qt.AlignCenter)
			layout.setContentsMargins(0, 0, 0, 0)
			cell_widget.setLayout(layout)
			self.tableWidget.setCellWidget(row, 1, cell_widget)
			
			# Color dropdown with colored items
			combo = QtWidgets.QComboBox()
			for color_name in self.available_colors:
				combo.addItem(color_name)
				# Set the background color of each item in the dropdown
				index = combo.count() - 1
				combo.setItemData(index, QColor(color_name), Qt.BackgroundRole)
				# Set text color to ensure visibility
				text_color = 'black' if color_name in ['yellow', 'cyan', 'lime', 'pink', 'coral', 'turquoise'] else 'white'
				combo.setItemData(index, QColor(text_color), Qt.ForegroundRole)
			
			# Set default color (use a different color for each row if possible)
			default_color_index = row % len(self.available_colors)
			combo.setCurrentIndex(default_color_index)
			
			# Update combo box appearance to match selected color
			color_name = self.available_colors[default_color_index]
			text_color = 'black' if color_name in ['yellow', 'cyan', 'lime', 'pink', 'coral', 'turquoise'] else 'white'
			combo.setStyleSheet(f"QComboBox {{ background-color: {color_name}; color: {text_color}; }}")
			
			# Connect signal to update appearance when selection changes
			combo.currentIndexChanged.connect(self.create_color_change_handler(combo))
			
			self.tableWidget.setCellWidget(row, 2, combo)
			
			# Transparency slider
			slider_widget = QtWidgets.QWidget()
			slider_layout = QtWidgets.QHBoxLayout(slider_widget)
			slider = QtWidgets.QSlider(Qt.Horizontal)
			slider.setMinimum(5)
			slider.setMaximum(50)  # Reduced maximum to ensure transparency
			slider.setValue(10)	 # Default 10% opacity (90% transparency) for better multi-view
			slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
			slider.setTickInterval(5)
			
			# Add value label
			value_label = QtWidgets.QLabel("10%")
			
			# Use a unique connection for each slider to avoid issues
			def make_update_func(label):
				return lambda val: label.setText(f"{val}%")
			
			slider.valueChanged.connect(make_update_func(value_label))
			
			slider_layout.addWidget(slider)
			slider_layout.addWidget(value_label)
			slider_layout.setContentsMargins(5, 0, 5, 0)
			slider_widget.setLayout(slider_layout)
			
			self.tableWidget.setCellWidget(row, 3, slider_widget)
	
	def create_color_change_handler(self, combo):
		"""Creates a handler for color changes in the combo box"""
		def handler(index):
			color_name = self.available_colors[index]
			# Update the combo box background to match the selected color
			text_color = 'black' if color_name in ['yellow', 'cyan', 'lime', 'pink', 'coral', 'turquoise'] else 'white'
			combo.setStyleSheet(f"QComboBox {{ background-color: {color_name}; color: {text_color}; }}")
		return handler
	
	def select_all(self):
		"""Selects all images in the table"""
		for row in range(self.tableWidget.rowCount()):
			cell_widget = self.tableWidget.cellWidget(row, 1)
			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
			checkbox.setChecked(True)
	
	def deselect_all(self):
		"""Deselects all images in the table"""
		for row in range(self.tableWidget.rowCount()):
			cell_widget = self.tableWidget.cellWidget(row, 1)
			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
			checkbox.setChecked(False)
	
	def get_selected_images(self):
		"""Gets the list of selected images and their chosen colors"""
		selected_images = []
		image_colors = {}
		image_transparency = {}
		
		for row in range(self.tableWidget.rowCount()):
			# Get checkbox state
			cell_widget = self.tableWidget.cellWidget(row, 1)
			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
			
			if checkbox.isChecked():
				# Get color choice
				combo = self.tableWidget.cellWidget(row, 2)
				color_index = combo.currentIndex()
				color = self.available_colors[color_index]
				
				# Get transparency value
				slider_widget = self.tableWidget.cellWidget(row, 3)
				slider = slider_widget.findChild(QtWidgets.QSlider)
				transparency = slider.value() / 100.0  # Convert to 0-1 range
				
				# Store selections
				selected_images.append(row)
				image_colors[row] = color
				image_transparency[row] = transparency
		
		return selected_images, image_colors, image_transparency
	
	def display_selected(self):
		"""Creates and displays a multi-phasor window with the selected images"""
		self.selected_images, self.image_colors, self.image_transparency = self.get_selected_images()
		# self.multi_phasor_window.set_lifetime_points((x_data, y_data))
		if not self.selected_images:
			# Show warning if no images selected
			QtWidgets.QMessageBox.warning(self, "No Selection", 
										 "Please select at least one image to display.")
			return
		
		# Create multi-phasor window if it doesn't exist yet
		if not self.multi_phasor_window or self.multi_phasor_window.dead:
			self.multi_phasor_window = MultiPhasorGraph("Multi-Phasor View")
		
		# Clear existing clouds
		for cloud_id in list(self.multi_phasor_window.phasor_clouds.keys()):
			self.multi_phasor_window.remove_phasor_cloud(cloud_id)
		
		# Add selected clouds with chosen colors and transparency
		for idx, image_idx in enumerate(self.selected_images):
			image = self.image_arr[image_idx]
			cloud_id = image.name
			
			# Get filtered g and s coordinates from the image
			# Create masks based on the thresholds applied in the individual image
			intensity_mask = image.intensity_mask
			plot_angle_mask = image.plot_angle_mask
			plot_circle_mask = image.plot_circle_mask
			plot_fraction_mask = image.plot_fraction_mask
			
			# Combine all masks
			combined_mask = intensity_mask | plot_angle_mask | plot_circle_mask | plot_fraction_mask
			combined_mask = combined_mask | (image.x_adjusted < 0)
			
			# Apply mask to get only visible points
			x_filtered = image.x_adjusted[~combined_mask].flatten()
			y_filtered = image.y_adjusted[~combined_mask].flatten()
			
			# Get selected color and transparency
			color = self.image_colors[image_idx]
			alpha = self.image_transparency[image_idx]
			
			# Add to multi-phasor view - pass color name directly
			self.multi_phasor_window.add_phasor_cloud(cloud_id, x_filtered, y_filtered, color, alpha)
		
		# Show the window
		self.multi_phasor_window.show()
