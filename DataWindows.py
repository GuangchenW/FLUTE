# Holds the classes to display the image, and to generate the matplotlib plot of the data, including the colormaps
# and the ranges specified by the user

#imports
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Qt5Agg')
import os
from matplotlib.image import NonUniformImage
import MplWidget

dir_path = os.path.dirname(os.path.realpath(__file__))

class Picture(QtWidgets.QMainWindow):
	"""Creates the picture window, and displays the images supplied by ImageHandler"""
	def __init__(self, name):
		super(Picture, self).__init__()

		height, width, channel = 512, 512, 3
		bytesPerLine = 3 * width

		self.widget = QLabel("HelloWorld")
		self.widget.setScaledContents(True)
		font = self.widget.font()
		font.setPointSize(30)
		self.widget.setFont(font)
		self.widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

		self.setCentralWidget(self.widget)
		self.name = name

		self.dead = False

	def set_image(self, im):
		"""Displays the image im"""
		im = cv2.resize(im, (512, 512))
		qImg = QImage(im.data, 512,512, QImage.Format_RGB888)
		self.widget.setPixmap(QPixmap(qImg))

	def closeEvent(self, event):
		"""Ran when the window is closed"""
		self.dead = True

	def set_window_number(self, num):
		"""Sets the title of the window to match the number in the table on the front panel"""
		self.setWindowTitle(str(num) +': ' + self.name)


class Calibration(QtWidgets.QMainWindow):
	"""Opens the calibration entry window to type in the lifetime"""
	def __init__(self):
		super(Calibration, self).__init__()
		uic.loadUi(dir_path + "/ui files/CalibrationWindow.ui", self)

class FLIMSelectionWindow(QtWidgets.QMainWindow):
	"""Instantiate the FLIM data selection window."""
	def __init__(self):
		super(FLIMSelectionWindow, self).__init__()
		uic.loadUi(dir_path + "/ui files/FLIMSelectionWindow.ui", self)

class CloseWindows(QtWidgets.QMainWindow):
	"""Opens the dialog to ask if the user really wants to close the windows"""
	def __init__(self):
		super(CloseWindows, self).__init__()
		uic.loadUi(dir_path + "/ui files/CloseWindow.ui", self)


class Fraction(QtWidgets.QMainWindow):
	"""Opens the fraction bound entry window"""
	def __init__(self):
		super(Fraction, self).__init__()
		uic.loadUi(dir_path + "/ui files/BoundEntry.ui", self)


class SaveWindow(QtWidgets.QMainWindow):
	"""Opens the window to enter what type of data to save. All or just current"""
	def __init__(self):
		super(SaveWindow, self).__init__()
		uic.loadUi(dir_path + "/ui files/SaveData.ui", self)


class Graph(QtWidgets.QMainWindow):
	"""Displays the MplWidget plot based on the thresholding parameters that the user enters"""
	def __init__(self, name, MHz):
		super(Graph, self).__init__()

		self.ui = uic.loadUi(dir_path + "/ui files/Graph.ui", self)

		x = np.linspace(0, 1, 1000)
		y = np.sqrt(0.5 * 0.5 - (x - 0.5) * (x - 0.5))
		self.Plot.canvas.ax.set_xlim([0, 1])
		self.Plot.canvas.ax.set_ylim([0, 0.6])
		self.Plot.canvas.ax.plot(x, y, 'r')
		self.Plot.canvas.ax.set_xlabel('g', fontsize=12, weight='bold')
		self.Plot.canvas.ax.set_ylabel('s', fontsize=12, weight='bold')

		self.MHz = '{0:.0f}'.format(MHz)

		# load the range lines horizontally and vertically
		y = np.tan((np.radians(0)) * x - 0.001)
		self.min_line, = self.Plot.canvas.ax.plot(x,y)

		y = np.tan(np.radians(90)) * x
		self.max_line, = self.Plot.canvas.ax.plot(x, y)

		self.min_circle, = self.Plot.canvas.ax.plot(x, y)
		self.max_circle, = self.Plot.canvas.ax.plot(x, y)

		self.circle_coors = np.full((4, 2),-3.0)
		self.circle_radius = [0.05, 0.05, 0.05, 0.05]

		self.circler = patches.Circle((-2, -2), 0.05, ec='r', alpha=0.7, lw=2.5)
		self.circleg = patches.Circle((-2, -2), 0.05, ec='g', alpha=0.7, lw=2.5)
		self.circleb = patches.Circle((-2, -2), 0.05, ec='b', alpha=0.7, lw=2.5)
		self.circley = patches.Circle((-2, -2), 0.05, ec='y', alpha=0.7, lw=2.5)

		self.Plot.canvas.ax.add_patch(self.circler)
		self.Plot.canvas.ax.add_patch(self.circleg)
		self.Plot.canvas.ax.add_patch(self.circleb)
		self.Plot.canvas.ax.add_patch(self.circley)

		self.circle_fraction_min = patches.Circle((0, 0), 0, ec='r', fill=0, lw=1.5)
		self.circle_fraction_max = patches.Circle((0, 0), 1.2, ec='r', fill=0, lw=1.5)

		self.Plot.canvas.ax.add_patch(self.circle_fraction_min)
		self.Plot.canvas.ax.add_patch(self.circle_fraction_max)

		self.name = name

		self.angle_min_val = 0
		self.angle_max_val = 90
		self.circle_min_val = 0
		self.circle_max_val = 120
		self.fraction_min = 0
		self.fraction_max = 1.2
		self.line_alpha = 1.0

		self.image_min_ang, self.image_max_ang = 0, 90
		self.image_min_M, self.image_max_M = 0, 120

		self.color_map = 0 #0=densitymap, 1=TauM, 2=TauP, 3=densitymap, 4=fractionBound

		self.cmap = matplotlib.cm.viridis.copy()
		self.cmap_r = matplotlib.cm.viridis_r.copy()
		self.cmap.set_bad('k', alpha=0)
		self.cmap_noir = matplotlib.cm.Greys.copy()

		self.x_fraction = 0
		self.y_fraction = 0

		self.dead = False

	def resizeEvent(self, event):
		self.Plot.setGeometry(0, 0, event.size().width(), event.size().height())

	def plot_data(self, x_data, y_data):
		"""Plots the xy data given by image handler, and colors based on the thresholds and colormap selected by the
		user"""
		# Placing the data into a histogram with reasonably sized binning helps speed up the plotting significantly
		H, xedges, yedges = np.histogram2d(x_data, y_data, bins=150, range=[[0, 1], [0, 0.6]])
		H = H.T
		xcenters = (xedges[:-1] + xedges[1:]) / 2
		ycenters = (yedges[:-1] + yedges[1:]) / 2
		x = np.tile(xcenters, (150,1))
		y = np.tile(ycenters, (150,1)).T
		# pre calculate distance D, fraction bound F, and angle A maps for the data
		D = np.sqrt(x**2+y**2)
		F = np.sqrt((x-self.x_fraction)**2+(y-self.y_fraction)**2)
		A = y/x
		min = np.tan(np.deg2rad(self.angle_min_val))
		max = np.tan(np.deg2rad(self.angle_max_val))
		# Convert the plot to an image, which makes it faster to plot than the raw data
		im = NonUniformImage(self.Plot.canvas.ax, interpolation='bilinear', cmap=self.cmap)
		im2 = NonUniformImage(self.Plot.canvas.ax, interpolation='bilinear', cmap=self.cmap_noir)
		im.set_data(xcenters, ycenters, A)
		# These if statements color the top image based on the thresholds, and then sets areas outside the thresholding
		# to be black.
		if self.color_map == 0:
			H = np.ma.masked_where(H < 0.005, H)
			im.set_data(xcenters, ycenters, H)
			H[H != 0] = 1
			im2.set_data(xcenters, ycenters, H)
		elif self.color_map == 1:
			im = NonUniformImage(self.Plot.canvas.ax, interpolation='bilinear', cmap=self.cmap_r)
			D = np.ma.masked_where((D < self.circle_min_val / 100) | (D > self.circle_max_val / 100) | (H < 0.01) |
								   (A < min) | (A > max) | (F<self.fraction_min) | (F>self.fraction_max),D)
			H[H != 0] = 1
			if not False in D.mask:
				D.mask[0, 0] = False
			im.set_data(xcenters, ycenters, D)
			im.set_clim(self.circle_min_val / 100, self.circle_max_val / 100)
			im2.set_data(xcenters, ycenters, H)
		elif self.color_map == 2:
			A[(A > self.image_max_ang) & (A < max)] = self.image_max_ang
			A[(A < self.image_min_ang) & (A > min)] = self.image_min_ang
			A = np.ma.masked_where((D < self.circle_min_val / 100) | (D > self.circle_max_val / 100) | (H < 0.01) |
								   (A < min) | (A > max) | (F<self.fraction_min) | (F>self.fraction_max),A)
			if not False in A.mask:
				A.mask[0, 0] = False
			im.set_data(xcenters, ycenters, A)
			im.set_clim(min, max)
			H[H != 0] = 1
			im2.set_data(xcenters, ycenters, H)
		elif self.color_map == 4:
			F = np.ma.masked_where((D < self.circle_min_val / 100) | (D > self.circle_max_val / 100) | (H < 0.01) |
								   (A < min) | (A > max) | (F<self.fraction_min) | (F>self.fraction_max),F)
			H[H != 0] = 1
			if not False in F.mask:
				F.mask[0, 0] = False
			im.set_data(xcenters, ycenters, F)
			im.set_clim(self.fraction_min, self.fraction_max)
			im2.set_data(xcenters, ycenters, H)
		for item in self.Plot.canvas.ax.get_images():
			item.remove()
		# Plot one image on top which has all the colours, and one image on the bottom which is just black to show the
		# points which are outside of the thresholding
		self.Plot.canvas.ax.add_image(im2)
		self.Plot.canvas.ax.add_image(im)
		if len(self.MHz) <= 2:
			self.Plot.canvas.ax.text(0.8, 0.55, self.MHz + " MHz", fontsize=12)
		else:
			self.Plot.canvas.ax.text(0.75, 0.55, self.MHz + " MHz", fontsize=12)
		# list = self.Plot.canvas.ax.get_images()


	def set_circle(self, selection):
		"""Changes the colour of the circle seleted for when the user clicks the plot based on the value in the
		enumerated dropdown box on the front panel"""
		self.circleSelect = selection

	def clear_circles(self):
		"""Sends all the circles to far outside the plot coordinates"""
		self.circle_coors[:] = -3.0
		self.draw_circles()

	def update_circle(self, event):
		"""Moves the selected circles"""
		self.circle_coors[self.circleSelect][0] = event.xdata
		self.circle_coors[self.circleSelect][1] = event.ydata
		self.draw_circles()

	def draw_circles(self):
		"""Draws the circles. Using patches as opposed to plotting them is far more efficient, otherwise the program
		hangs for a while"""
		self.circler.remove()
		self.circler = patches.Circle((self.circle_coors[0][0], self.circle_coors[0][1]), self.circle_radius[0],
									  ec='r', fill=0, alpha=0.7, lw=2.5)
		self.Plot.canvas.ax.add_patch(self.circler)

		self.circleg.remove()
		self.circleg = patches.Circle((self.circle_coors[1][0], self.circle_coors[1][1]), self.circle_radius[1],
									  ec='g', fill=0, alpha=0.7, lw=2.5)
		self.Plot.canvas.ax.add_patch(self.circleg)

		self.circleb.remove()
		self.circleb = patches.Circle((self.circle_coors[2][0], self.circle_coors[2][1]), self.circle_radius[2],
									  ec='b', fill=0, alpha=0.7, lw=2.5)
		self.Plot.canvas.ax.add_patch(self.circleb)

		self.circley.remove()
		self.circley = patches.Circle((self.circle_coors[3][0], self.circle_coors[3][1]), self.circle_radius[3],
									  ec='y', fill=0, alpha=0.7, lw=2.5)
		self.Plot.canvas.ax.add_patch(self.circley)

		self.Plot.canvas.draw()

	def update_fraction_range(self, min, max, *args, **kwargs):
		"""Draws the circles for fraction range. Need to clear plot and then redraw it. This is why it's far more
		efficient to work with images rather than the raw histogram data"""
		self.fraction_min = min
		self.fraction_max = max
		self.circle_fraction_min.remove()
		self.circle_fraction_max.remove()
		self.circle_fraction_min = patches.Circle((self.x_fraction, self.y_fraction), min, ec='b', fill=0, lw=1.5, alpha = self.line_alpha)
		self.circle_fraction_max = patches.Circle((self.x_fraction, self.y_fraction), max, ec='b', fill=0, lw=1.5, alpha = self.line_alpha)
		self.Plot.canvas.ax.add_patch(self.circle_fraction_min)
		self.Plot.canvas.ax.add_patch(self.circle_fraction_max)
		self.Plot.canvas.draw()

	def update_angle_range(self, min, max, *args, **kwargs):
		"""Draws the lines for angle range. Need to clear plot and then redraw it. This is why it's far more
		efficient to work with images rather than the raw histogram data"""
		self.min_line.remove()
		self.max_line.remove()
		x = np.linspace(0,2,3)
		y = np.tan((np.deg2rad(min)))*x
		if y[-1] == 0:
			y = [-1, -1, -1]

		self.min_line, = self.Plot.canvas.ax.plot(x, y, color='r', alpha = self.line_alpha)

		y = np.tan(np.radians(max))*x
		self.max_line, = self.Plot.canvas.ax.plot(x, y, color = 'r', alpha = self.line_alpha)

		self.angle_min_val = min
		self.angle_max_val = max

	def update_circle_range(self, min, max, *args, **kwargs):
		"""Draws the circles for modulation range. Need to clear plot and then redraw it. This is why it's far more
		efficient to work with images rather than the raw histogram data"""
		self.min_circle.remove()
		self.max_circle.remove()

		x1 = np.linspace(0, min/100, 100)
		y1 = np.sqrt((min/100)**2 - x1**2)
		self.min_circle, = self.Plot.canvas.ax.plot(x1, y1, color='r', alpha = self.line_alpha)

		x2 = np.linspace(0, max/100, 100)
		y2 = np.sqrt((max/100)**2 - x2**2)
		self.max_circle, = self.Plot.canvas.ax.plot(x2, y2, color='r', alpha = self.line_alpha)

		self.circle_min_val = min
		self.circle_max_val = max

	def change_circle_radius(self, radius):
		"""Makes the click circles of radius = radius"""
		self.circle_radius[self.circleSelect] = radius
		self.draw_circles()

	def closeEvent(self, event):
		"""Ran when the window is closed"""
		self.dead = True

	def update_data(self, x, y, col_map = 0):
		"""plots new data, and adds the thresholding lines and circles to a new plot"""
		self.plot_data(x.flatten(),y.flatten())
		self.update_angle_range(self.angle_min_val, self.angle_max_val)
		self.update_circle_range(self.circle_min_val, self.circle_max_val)
		self.draw_circles()

	def set_window_number(self, num):
		"""Sets the title of the window"""
		self.setWindowTitle(str(num) +': ' + self.name)

	def set_colormap(self, val):
		"""updates the colormap value"""
		self.color_map = val

	def set_image_props(self, min_ang, max_ang, min_m, max_m):
		"""Changes the thresholding parameters for angle and modulation"""
		self.image_min_ang = min_ang
		self.image_max_ang = max_ang
		self.image_min_M = min_m
		self.image_max_M = max_m

	def set_lifetime_points(self, *args):
		"""Adds the lifetime values to the universal circle"""
		lifetime_x = args[0][0]
		lifetime_y = args[0][1]
		lifetimes = [0.5, 1, 2, 3, 4, 8]
		self.Plot.canvas.ax.scatter(lifetime_x, lifetime_y, color='r', s=10)
		for i in range(6):
			self.Plot.canvas.ax.text(lifetime_x[i]-0.05, lifetime_y[i]+0.03, str(lifetimes[i]) + " ns", color='r', fontsize=9)

	def set_fraction(self, x, y):
		"""Changes the thresholding parameters for the fraction bound circles"""
		self.x_fraction = x
		self.y_fraction = y

	def save_fig(self, file):
		"""Saves a picture of the plot in file path"""
		self.Plot.canvas.save_fig(file)

	def set_alpha(self, value):
		self.line_alpha = value
		self.update_circle_range(self.circle_min_val, self.circle_max_val)
		self.update_angle_range(self.angle_min_val, self.angle_max_val)
		self.update_fraction_range(self.fraction_min, self.fraction_max)

# class MultiPhasorGraph(QtWidgets.QMainWindow):
# 	"""Displays multiple phasor clouds on a single plot with customizable colormaps"""
# 	def __init__(self, name):
# 		super(MultiPhasorGraph, self).__init__()

# 		self.ui = uic.loadUi(dir_path + "/ui files/Graph.ui", self)

# 		x = np.linspace(0, 1, 1000)
# 		y = np.sqrt(0.5 * 0.5 - (x - 0.5) * (x - 0.5))
# 		self.Plot.canvas.ax.set_xlim([0, 1])
# 		self.Plot.canvas.ax.set_ylim([0, 0.6])
# 		self.Plot.canvas.ax.plot(x, y, 'r')
# 		self.Plot.canvas.ax.set_xlabel('g', fontsize=12, weight='bold')
# 		self.Plot.canvas.ax.set_ylabel('s', fontsize=12, weight='bold')

# 		# Dictionary to store multiple phasor clouds
# 		self.phasor_clouds = {}
# 		# Dictionary to store the colormap for each phasor cloud
# 		self.cloud_cmaps = {}
# 		# Dictionary to store transparency for each cloud
# 		self.cloud_alphas = {}
		
# 		# Available colormaps
# 		self.available_cmaps = {
# 			'viridis': matplotlib.cm.viridis.copy(),
# 			'viridis_r': matplotlib.cm.viridis_r.copy(),
# 			'plasma': matplotlib.cm.plasma.copy(),
# 			'inferno': matplotlib.cm.inferno.copy(),
# 			'magma': matplotlib.cm.magma.copy(),
# 			'cividis': matplotlib.cm.cividis.copy(),
# 			'jet': matplotlib.cm.jet.copy(),
# 			'rainbow': matplotlib.cm.rainbow.copy(),
# 			'Greys': matplotlib.cm.Greys.copy(),
# 			'hot': matplotlib.cm.hot.copy(),
# 			'cool': matplotlib.cm.cool.copy()
# 		}
# 		# self.color_palette = {
# 		# 	'primary': '#CBE330',
# 		# 	'secondary': '#35E330',
# 		# 	'tertiary': '#80E330',
# 		# 	'accent1': '#E3D430',
# 		# 	'accent2': '#30E368'
# 		# }
		
# 		# Set transparent background for all colormaps
# 		for cmap in self.available_cmaps.values():
# 			cmap.set_bad('k', alpha=0)
			
# 		self.name = name
# 		self.dead = False

# 	def resizeEvent(self, event):
# 		self.Plot.setGeometry(0, 0, event.size().width(), event.size().height())

# 	def add_phasor_cloud(self, cloud_id, x_data, y_data, cmap_name='viridis', alpha=0.7):
# 		"""Adds a phasor cloud to the collection with a specified colormap and transparency"""
# 		if cmap_name not in self.available_cmaps:
# 			cmap_name = 'viridis'
			
# 		# Store the data
# 		self.phasor_clouds[cloud_id] = (x_data, y_data)
# 		# Store the colormap
# 		self.cloud_cmaps[cloud_id] = cmap_name
# 		# Store the transparency
# 		self.cloud_alphas[cloud_id] = alpha
# 		# Replot everything
# 		self.plot_all_clouds()
		
# 	def remove_phasor_cloud(self, cloud_id):
# 		"""Removes a phasor cloud from the collection"""
# 		if cloud_id in self.phasor_clouds:
# 			del self.phasor_clouds[cloud_id]
# 			del self.cloud_cmaps[cloud_id]
# 			if cloud_id in self.cloud_alphas:
# 				del self.cloud_alphas[cloud_id]
# 			# Replot everything
# 			self.plot_all_clouds()
			
# 	def set_cloud_colormap(self, cloud_id, cmap_name):
# 		"""Changes the colormap for a specific phasor cloud"""
# 		if cloud_id in self.cloud_cmaps and cmap_name in self.available_cmaps:
# 			self.cloud_cmaps[cloud_id] = cmap_name
# 			# Replot everything
# 			self.plot_all_clouds()
			
# 	def set_cloud_transparency(self, cloud_id, alpha):
# 		"""Changes the transparency for a specific phasor cloud"""
# 		if cloud_id in self.phasor_clouds:
# 			self.cloud_alphas[cloud_id] = alpha
# 			# Replot everything
# 			self.plot_all_clouds()
	
# 	def get_available_colormaps(self):
# 		"""Returns a list of available colormap names"""
# 		return list(self.available_cmaps.keys())
			
# 	def plot_all_clouds(self):
# 		"""Plots all phasor clouds with their respective colormaps and transparency"""
# 		# Clear current images and scatter plots
# 		for item in self.Plot.canvas.ax.get_images():
# 			item.remove()
		
# 		# Also clear any scatter plots
# 		for artist in self.Plot.canvas.ax.collections:
# 			artist.remove()
			
# 		# Plot each cloud with scatter plots for better transparency
# 		for idx, (cloud_id, (x_data, y_data)) in enumerate(self.phasor_clouds.items()):
# 			cmap_name = self.cloud_cmaps[cloud_id]
			
# 			# Get transparency for this cloud (default to 0.3 if not set)
# 			alpha = self.cloud_alphas.get(cloud_id, 0.3)
			
# 			# Get a representative color from the colormap
# 			color = self.available_cmaps[cmap_name](0.8)[:3]
			
# 			# Use scatter plot with decimation for better performance
# 			# Only plot a subset of points if there are too many
# 			max_points = 10000  # Reasonable limit for plot performance
			
# 			if len(x_data) > max_points:
# 				# Randomly sample points to avoid performance issues
# 				import random
# 				indices = random.sample(range(len(x_data)), max_points)
# 				x_plot = [x_data[i] for i in indices]
# 				y_plot = [y_data[i] for i in indices]
# 			else:
# 				x_plot = x_data
# 				y_plot = y_data
				
# 			# Use scatter plot with smaller point size for better transparency and visibility
# 			self.Plot.canvas.ax.scatter(
# 				x_plot, y_plot, 
# 				color=color, 
# 				alpha=alpha,
# 				s=10,  # Small point size
# 				marker='.',
# 				linewidths=0,
# 				edgecolors='none'
# 			)
# 			mean_x = np.nanmean(x_plot)  # Using nanmean to ignore NaN values
# 			mean_y = np.nanmean(y_plot)  # Using nanmean to ignore NaN values

# 			# Plot the center point with a distinctive appearance
# 			self.Plot.canvas.ax.scatter(
# 				mean_x, mean_y,
# 				color=color,  # Choose a contrasting color
# 				s=100,  # Larger point size for visibility
# 				marker='X',  # Distinctive marker
# 				edgecolors='black',
# 				linewidths=1.5,
# 				zorder=10,  # Ensure it's drawn on top
# 				label=f'Center ({mean_x:.2f}, {mean_y:.2f})'
# 			)
			
# 		# Add a legend with colormap names and transparency
# 		self.add_legend()
# 		self.Plot.canvas.draw()
		
# 	def add_legend(self):
# 		"""Adds a legend showing which colormap is used for each cloud"""
# 		# Create custom legend entries
# 		import matplotlib.patches as mpatches
# 		legend_entries = []
		
# 		for cloud_id, cmap_name in self.cloud_cmaps.items():
# 			alpha = self.cloud_alphas.get(cloud_id, 0.7)
# 			color = self.available_cmaps[cmap_name](0.8)[:3]  # Get a representative color
# 			patch = mpatches.Patch(color=color, alpha=alpha, label=f"{cloud_id} ({cmap_name}, {int(alpha*100)}%)")
# 			legend_entries.append(patch)
			
# 		# Add the legend
# 		if legend_entries:
# 			self.Plot.canvas.ax.legend(handles=legend_entries, loc='upper right', fontsize='small')
			
# 	def closeEvent(self, event):
# 		"""Ran when the window is closed"""
# 		self.dead = True
		
# 	def save_fig(self, file):
# 		"""Saves the figure as a png file"""
# 		self.Plot.canvas.fig.savefig(file)

# class MultiPhasorSelector(QtWidgets.QMainWindow):
# 	"""Dialog for selecting which phasor clouds to display in the multi-view and their colormaps"""
# 	def __init__(self, image_arr):
# 		super(MultiPhasorSelector, self).__init__()
# 		uic.loadUi(dir_path + "/ui files/MultiPhasorSelector.ui", self)
		
# 		self.image_arr = image_arr
# 		self.selected_images = []
# 		self.image_colormaps = {}
# 		self.image_transparency = {}
		
# 		# Connect signals
# 		self.btnSelectAll.clicked.connect(self.select_all)
# 		self.btnDeselectAll.clicked.connect(self.deselect_all)
# 		self.btnDisplay.clicked.connect(self.display_selected)
# 		self.btnClose.clicked.connect(self.close)
		
# 		# Store available colormaps for dropdown
# 		self.available_cmaps = [
# 			'viridis', 'viridis_r', 'plasma', 'inferno', 'magma', 
# 			'cividis', 'jet', 'rainbow', 'Greys', 'hot', 'cool'
# 		]
	
		
		
# 		# Set up table
# 		self.populate_table()
		
# 		self.multi_phasor_window = None
		
# 	def populate_table(self):
# 		"""Populates the table with loaded images"""
# 		self.tableWidget.setRowCount(len(self.image_arr))
		
# 		for row, image in enumerate(self.image_arr):
# 			# Image name
# 			name_item = QtWidgets.QTableWidgetItem(image.name)
# 			name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
# 			self.tableWidget.setItem(row, 0, name_item)
			
# 			# Checkbox for inclusion
# 			checkbox = QtWidgets.QCheckBox()
# 			cell_widget = QtWidgets.QWidget()
# 			layout = QtWidgets.QHBoxLayout(cell_widget)
# 			layout.addWidget(checkbox)
# 			layout.setAlignment(Qt.AlignCenter)
# 			layout.setContentsMargins(0, 0, 0, 0)
# 			cell_widget.setLayout(layout)
# 			self.tableWidget.setCellWidget(row, 1, cell_widget)
			
# 			# Colormap dropdown
# 			combo = QtWidgets.QComboBox()
# 			combo.addItems(self.available_cmaps)
# 			self.tableWidget.setCellWidget(row, 2, combo)
			
# 			# Transparency slider
# 			slider_widget = QtWidgets.QWidget()
# 			slider_layout = QtWidgets.QHBoxLayout(slider_widget)
# 			slider = QtWidgets.QSlider(Qt.Horizontal)
# 			slider.setMinimum(5)
# 			slider.setMaximum(50)  # Reduced maximum to ensure transparency
# 			slider.setValue(10)  # Default 10% opacity (90% transparency) for better multi-view
# 			slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
# 			slider.setTickInterval(5)
			
# 			# Add value label
# 			value_label = QtWidgets.QLabel("10%")
			
# 			# Use a unique connection for each slider to avoid issues
# 			def make_update_func(label):
# 				return lambda val: label.setText(f"{val}%")
			
# 			slider.valueChanged.connect(make_update_func(value_label))
			
# 			slider_layout.addWidget(slider)
# 			slider_layout.addWidget(value_label)
# 			slider_layout.setContentsMargins(5, 0, 5, 0)
# 			slider_widget.setLayout(slider_layout)
			
# 			self.tableWidget.setCellWidget(row, 3, slider_widget)
	
# 	def select_all(self):
# 		"""Selects all images in the table"""
# 		for row in range(self.tableWidget.rowCount()):
# 			cell_widget = self.tableWidget.cellWidget(row, 1)
# 			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
# 			checkbox.setChecked(True)
	
# 	def deselect_all(self):
# 		"""Deselects all images in the table"""
# 		for row in range(self.tableWidget.rowCount()):
# 			cell_widget = self.tableWidget.cellWidget(row, 1)
# 			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
# 			checkbox.setChecked(False)
	
# 	def get_selected_images(self):
# 		"""Gets the list of selected images and their chosen colormaps"""
# 		selected_images = []
# 		image_colormaps = {}
# 		image_transparency = {}
		
# 		for row in range(self.tableWidget.rowCount()):
# 			# Get checkbox state
# 			cell_widget = self.tableWidget.cellWidget(row, 1)
# 			checkbox = cell_widget.findChild(QtWidgets.QCheckBox)
			
# 			if checkbox.isChecked():
# 				# Get colormap choice
# 				combo = self.tableWidget.cellWidget(row, 2)
# 				colormap = combo.currentText()
				
# 				# Get transparency value
# 				slider_widget = self.tableWidget.cellWidget(row, 3)
# 				slider = slider_widget.findChild(QtWidgets.QSlider)
# 				transparency = slider.value() / 100.0  # Convert to 0-1 range
				
# 				# Store selections
# 				selected_images.append(row)
# 				image_colormaps[row] = colormap
# 				image_transparency[row] = transparency
		
# 		return selected_images, image_colormaps, image_transparency
	
# 	def display_selected(self):
# 		"""Creates and displays a multi-phasor window with the selected images"""
# 		self.selected_images, self.image_colormaps, self.image_transparency = self.get_selected_images()
		
# 		if not self.selected_images:
# 			# Show warning if no images selected
# 			QtWidgets.QMessageBox.warning(self, "No Selection", 
# 										 "Please select at least one image to display.")
# 			return
		
# 		# Create multi-phasor window if it doesn't exist yet
# 		if not self.multi_phasor_window or self.multi_phasor_window.dead:
# 			self.multi_phasor_window = MultiPhasorGraph("Multi-Phasor View")
		
# 		# Clear existing clouds
# 		for cloud_id in list(self.multi_phasor_window.phasor_clouds.keys()):
# 			self.multi_phasor_window.remove_phasor_cloud(cloud_id)
		
# 		# Add selected clouds with chosen colormaps and transparency
# 		for idx, image_idx in enumerate(self.selected_images):
# 			image = self.image_arr[image_idx]
# 			cloud_id = image.name
			
# 			# Get filtered g and s coordinates from the image
# 			# Create masks based on the thresholds applied in the individual image
# 			intensity_mask = image.intensity_mask
# 			plot_angle_mask = image.plot_angle_mask
# 			plot_circle_mask = image.plot_circle_mask
# 			plot_fraction_mask = image.plot_fraction_mask
			
# 			# Combine all masks
# 			combined_mask = intensity_mask | plot_angle_mask | plot_circle_mask | plot_fraction_mask
# 			combined_mask = combined_mask | (image.x_adjusted < 0)
			
# 			# Apply mask to get only visible points
# 			x_filtered = image.x_adjusted[~combined_mask].flatten()
# 			y_filtered = image.y_adjusted[~combined_mask].flatten()
			
# 			# Get selected colormap and transparency
# 			cmap = self.image_colormaps[image_idx]
# 			alpha = self.image_transparency[image_idx]
			
# 			# Add to multi-phasor view
# 			self.multi_phasor_window.add_phasor_cloud(cloud_id, x_filtered, y_filtered, cmap, alpha)
		
# 		# Show the window
# 		self.multi_phasor_window.show()

class MultiPhasorGraph(QtWidgets.QMainWindow):
    """Displays multiple phasor clouds on a single plot with customizable colors or colormaps"""
    def __init__(self, name):
        super(MultiPhasorGraph, self).__init__()

        self.ui = uic.loadUi(dir_path + "/ui files/Graph.ui", self)

        x = np.linspace(0, 1, 1000)
        y = np.sqrt(0.5 * 0.5 - (x - 0.5) * (x - 0.5))
        self.Plot.canvas.ax.set_xlim([0, 1])
        self.Plot.canvas.ax.set_ylim([0, 0.6])
        self.Plot.canvas.ax.plot(x, y, 'r')
        self.Plot.canvas.ax.set_xlabel('g', fontsize=12, weight='bold')
        self.Plot.canvas.ax.set_ylabel('s', fontsize=12, weight='bold')

        # Dictionary to store multiple phasor clouds
        self.phasor_clouds = {}
        # Dictionary to store the color or colormap for each phasor cloud
        self.cloud_cmaps = {}  # Keeping the name for backward compatibility
        # Dictionary to store transparency for each cloud
        self.cloud_alphas = {}
        
        # # Available colormaps
        # self.available_cmaps = {
        #     'viridis': matplotlib.cm.viridis.copy(),
        #     'viridis_r': matplotlib.cm.viridis_r.copy(),
        #     'plasma': matplotlib.cm.plasma.copy(),
        #     'inferno': matplotlib.cm.inferno.copy(),
        #     'magma': matplotlib.cm.magma.copy(),
        #     'cividis': matplotlib.cm.cividis.copy(),
        #     'jet': matplotlib.cm.jet.copy(),
        #     'rainbow': matplotlib.cm.rainbow.copy(),
        #     'Greys': matplotlib.cm.Greys.copy(),
        #     'hot': matplotlib.cm.hot.copy(),
        #     'cool': matplotlib.cm.cool.copy()
        # }
        
        # Available direct colors (new addition)
        self.available_colors = [
            'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 
            'orange', 'purple', 'lime', 'pink', 'brown', 'navy',
            'teal', 'olive', 'maroon', 'coral', 'indigo', 'turquoise'
        ]
        
        # # Set transparent background for all colormaps
        # for cmap in self.available_cmaps.values():
        #     cmap.set_bad('k', alpha=0)
            
        self.name = name
        self.dead = False

    def resizeEvent(self, event):
        self.Plot.setGeometry(0, 0, event.size().width(), event.size().height())

    def add_phasor_cloud(self, cloud_id, x_data, y_data, color_or_cmap='red', alpha=0.7):
        """Adds a phasor cloud to the collection with a specified color/colormap and transparency"""
        # Store the data
        self.phasor_clouds[cloud_id] = (x_data, y_data)
        # Store the color/colormap
        self.cloud_cmaps[cloud_id] = color_or_cmap
        # Store the transparency
        self.cloud_alphas[cloud_id] = alpha
        # Replot everything
        self.plot_all_clouds()
        
    def remove_phasor_cloud(self, cloud_id):
        """Removes a phasor cloud from the collection"""
        if cloud_id in self.phasor_clouds:
            del self.phasor_clouds[cloud_id]
            del self.cloud_cmaps[cloud_id]
            if cloud_id in self.cloud_alphas:
                del self.cloud_alphas[cloud_id]
            # Replot everything
            self.plot_all_clouds()
            
    def set_cloud_colormap(self, cloud_id, color_or_cmap):
        """Changes the color/colormap for a specific phasor cloud"""
        if cloud_id in self.cloud_cmaps:
            self.cloud_cmaps[cloud_id] = color_or_cmap
            # Replot everything
            self.plot_all_clouds()
            
    def set_cloud_transparency(self, cloud_id, alpha):
        """Changes the transparency for a specific phasor cloud"""
        if cloud_id in self.phasor_clouds:
            self.cloud_alphas[cloud_id] = alpha
            # Replot everything
            self.plot_all_clouds()
    
    # def get_available_colormaps(self):
    #     """Returns a list of available colormap names"""
    #     return list(self.available_cmaps.keys())
    
    def get_available_colors(self):
        """Returns a list of available color names"""
        return self.available_colors
            
    def plot_all_clouds(self):
        """Plots all phasor clouds with their respective colors/colormaps and transparency"""
        # Clear current images and scatter plots
        for item in self.Plot.canvas.ax.get_images():
            item.remove()
        
        # Also clear any scatter plots
        for artist in self.Plot.canvas.ax.collections:
            artist.remove()
            
        # Plot each cloud with scatter plots for better transparency
        for idx, (cloud_id, (x_data, y_data)) in enumerate(self.phasor_clouds.items()):
            color_or_cmap = self.cloud_cmaps[cloud_id]
            
            # Get transparency for this cloud (default to 0.3 if not set)
            alpha = self.cloud_alphas.get(cloud_id, 0.3)
            
            # # Determine if it's a colormap or direct color
            # if color_or_cmap in self.available_cmaps:
            #     # It's a colormap
            #     color = self.available_cmaps[color_or_cmap](0.8)[:3]
            # else:
            #     # It's a direct color
            #     color = color_or_cmap
            color = color_or_cmap
            # Use scatter plot with decimation for better performance
            # Only plot a subset of points if there are too many
            max_points = 10000  # Reasonable limit for plot performance
            
            if len(x_data) > max_points:
                # Randomly sample points to avoid performance issues
                import random
                indices = random.sample(range(len(x_data)), max_points)
                x_plot = [x_data[i] for i in indices]
                y_plot = [y_data[i] for i in indices]
            else:
                x_plot = x_data
                y_plot = y_data
                
            # Use scatter plot with smaller point size for better transparency and visibility
            self.Plot.canvas.ax.scatter(
                x_plot, y_plot, 
                color=color, 
                alpha=alpha,
                s=10.0,  # Smaller point size (3)
                marker='.',
                linewidths=0,
                edgecolors='none'
            )
            
            # Calculate and plot center point
            mean_x = np.nanmean(x_plot)  # Using nanmean to ignore NaN values
            mean_y = np.nanmean(y_plot)  # Using nanmean to ignore NaN values

            # Plot the center point with a distinctive appearance
            self.Plot.canvas.ax.scatter(
                mean_x, mean_y,
                color=color,  # Same color as the cloud
                s=100,  # Larger point size for visibility
                marker='X',  # Distinctive marker
                edgecolors='black',
                linewidths=1.5,
                zorder=10,  # Ensure it's drawn on top
                label=f'Center ({mean_x:.2f}, {mean_y:.2f})'
            )
            
        # Add a legend with colormap/color names and transparency
        self.add_legend()
        self.Plot.canvas.draw()
        
    def add_legend(self):
        """Adds a legend showing which color/colormap is used for each cloud"""
        # Create custom legend entries
        import matplotlib.patches as mpatches
        legend_entries = []
        
        for cloud_id, color_or_cmap in self.cloud_cmaps.items():
            alpha = self.cloud_alphas.get(cloud_id, 0.7)
            
            # # Determine if it's a colormap or direct color
            # if color_or_cmap in self.available_cmaps:
            #     # It's a colormap
            #     color = self.available_cmaps[color_or_cmap](0.8)[:3]
            #     label = f"{cloud_id} ({color_or_cmap}, {int(alpha*100)}%)"
            # else:
            #     # It's a direct color
            #     color = color_or_cmap
            #     label = f"{cloud_id} ({color}, {int(alpha*100)}%)"
            color = color_or_cmap
            label = f"{cloud_id} ({color}, {int(alpha*100)}%)"
            
            patch = mpatches.Patch(color=color, alpha=alpha, label=label)
            legend_entries.append(patch)
            
        # Add the legend
        if legend_entries:
            self.Plot.canvas.ax.legend(handles=legend_entries, loc='upper right', fontsize='small')
            
    def closeEvent(self, event):
        """Ran when the window is closed"""
        self.dead = True
        
    def save_fig(self, file):
        """Saves the figure as a png file"""
        self.Plot.canvas.fig.savefig(file)


class MultiPhasorSelector(QtWidgets.QMainWindow):
    """Dialog for selecting which phasor clouds to display in the multi-view and their colors"""
    def __init__(self, image_arr):
        super(MultiPhasorSelector, self).__init__()
        uic.loadUi(dir_path + "/ui files/MultiPhasorSelector.ui", self)
        
        self.image_arr = image_arr
        self.selected_images = []
        self.image_colors = {}  # Changed from image_colormaps
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
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
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
            slider.setValue(10)  # Default 10% opacity (90% transparency) for better multi-view
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

