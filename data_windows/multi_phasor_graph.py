#imports
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QLabel, QFileDialog
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Qt5Agg')

from config import UI_DIR

class MultiPhasorGraph(QtWidgets.QMainWindow):
	"""Displays multiple phasor clouds on a single plot with customizable colors or colormaps"""
	def __init__(self, name):
		super(MultiPhasorGraph, self).__init__()
		
		# Load the ui component
		ui_path = UI_DIR / "Graph.ui"
		uic.loadUi(str(ui_path), self)
		
		self.btnSavePlot.clicked.connect(self.save_fig)
		
		x = np.linspace(0, 1, 1000)
		y = np.sqrt(0.5 * 0.5 - (x - 0.5) * (x - 0.5))
		self.Plot.canvas.ax.set_xlim([0, 1])
		self.Plot.canvas.ax.set_ylim([0, 0.6])
		self.Plot.canvas.ax.plot(x, y, 'r')
		self.Plot.canvas.ax.set_xlabel('g', fontsize=12, weight='bold')
		self.Plot.canvas.ax.set_ylabel('s', fontsize=12, weight='bold')
		
		# Add lifetime labels on the semicircle
		MHz = 80  # or use the actual MHz value if available
		lifetimes = [0.5, 1, 2, 3, 4, 8]  # ns
		omega = 2 * np.pi * MHz * 1e6
		tau = np.array(lifetimes) * 1e-9
		g = 1 / (1 + (omega * tau) ** 2)
		s = (omega * tau) / (1 + (omega * tau) ** 2)
		# Normalize to fit the semicircle (as in your plot)
		# # Place the labels
		# for i, t in enumerate(lifetimes):
		#	  self.Plot.canvas.ax.text(g[i]-0.05, s[i]+0.03, f"{t} ns", color='r', fontsize=9)
		print(g, s)	   
		# Plot the dots
		self.Plot.canvas.ax.scatter(g, s, color='r', s=10)
		print(g, s)
		# Place the labels next to the dots
		for i, t in enumerate(lifetimes):
			self.Plot.canvas.ax.text(g[i]-0.04, s[i]+0.03, f"{t} ns", color='r', fontsize=9)
		self.Plot.canvas.draw()
		

		# Dictionary to store multiple phasor clouds
		self.phasor_clouds = {}
		# Dictionary to store the color or colormap for each phasor cloud
		self.cloud_cmaps = {}  # Keeping the name for backward compatibility
		# Dictionary to store transparency for each cloud
		self.cloud_alphas = {}
		
		# # Available colormaps
		# self.available_cmaps = {
		#	  'viridis': matplotlib.cm.viridis.copy(),
		#	  'viridis_r': matplotlib.cm.viridis_r.copy(),
		#	  'plasma': matplotlib.cm.plasma.copy(),
		#	  'inferno': matplotlib.cm.inferno.copy(),
		#	  'magma': matplotlib.cm.magma.copy(),
		#	  'cividis': matplotlib.cm.cividis.copy(),
		#	  'jet': matplotlib.cm.jet.copy(),
		#	  'rainbow': matplotlib.cm.rainbow.copy(),
		#	  'Greys': matplotlib.cm.Greys.copy(),
		#	  'hot': matplotlib.cm.hot.copy(),
		#	  'cool': matplotlib.cm.cool.copy()
		# }
		
		# Available direct colors (new addition)
		self.available_colors = [
			'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 
			'orange', 'purple', 'lime', 'pink', 'brown', 'navy',
			'teal', 'olive', 'maroon', 'coral', 'indigo', 'turquoise'
		]
		
		# # Set transparent background for all colormaps
		# for cmap in self.available_cmaps.values():
		#	  cmap.set_bad('k', alpha=0)
			
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
	#	  """Returns a list of available colormap names"""
	#	  return list(self.available_cmaps.keys())
	
	def get_available_colors(self):
		"""Returns a list of available color names"""
		return self.available_colors
	def set_lifetime_points(self, *args):
		"""Adds the lifetime values to the universal circle"""
		lifetime_x = args[0][0]
		lifetime_y = args[0][1]
		lifetimes = [0.5, 1, 2, 3, 4, 8]
		self.Plot.canvas.ax.scatter(lifetime_x, lifetime_y, color='r', s=10)
		for i in range(6):
			self.Plot.canvas.ax.text(lifetime_x[i]-0.05, lifetime_y[i]+0.03, str(lifetimes[i]) + " ns", color='r', fontsize=9)
			
	def plot_all_clouds(self):
		"""Plots all phasor clouds with their respective colors/colormaps and transparency"""
		# Clear current images and scatter plots
		for item in self.Plot.canvas.ax.get_images():
			item.remove()
		
		# # Also clear any scatter plots
		# for artist in self.Plot.canvas.ax.collections:
		#	  artist.remove()
			
		# Plot each cloud with scatter plots for better transparency
		for idx, (cloud_id, (x_data, y_data)) in enumerate(self.phasor_clouds.items()):
			color_or_cmap = self.cloud_cmaps[cloud_id]
			
			# Get transparency for this cloud (default to 0.3 if not set)
			alpha = self.cloud_alphas.get(cloud_id, 0.3)
			
			# # Determine if it's a colormap or direct color
			# if color_or_cmap in self.available_cmaps:
			#	  # It's a colormap
			#	  color = self.available_cmaps[color_or_cmap](0.8)[:3]
			# else:
			#	  # It's a direct color
			#	  color = color_or_cmap
			color = color_or_cmap
			# Use scatter plot with decimation for better performance
			# Only plot a subset of points if there are too many
			max_points = 10000	# Reasonable limit for plot performance
			
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
				s=10.0,	 # Smaller point size (3)
				marker='.',
				linewidths=0,
				edgecolors='none'
			)
			
			# Calculate and plot center point
			mean_x = np.nanmean(x_plot)	 # Using nanmean to ignore NaN values
			mean_y = np.nanmean(y_plot)	 # Using nanmean to ignore NaN values

			# Plot the center point with a distinctive appearance
			self.Plot.canvas.ax.scatter(
				mean_x, mean_y,
				color=color,  # Same color as the cloud
				s=100,	# Larger point size for visibility
				marker='X',	 # Distinctive marker
				edgecolors='black',
				linewidths=1.5,
				zorder=10,	# Ensure it's drawn on top
				label=f'Center ({mean_x:.2f}, {mean_y:.2f})'
			)
		# self.set_lifetime_points(x_data[0],y_data[0])		
		# Add a legend with colormap/color names and transparency
		self.add_legend()
		self.Plot.canvas.draw()
		
	def add_legend(self):
		"""Adds a legend showing which color/colormap is used for each cloud"""
		# Create custom legend entries
		legend_entries = []
		
		for cloud_id, color_or_cmap in self.cloud_cmaps.items():
			alpha = self.cloud_alphas.get(cloud_id, 0.7)
			
			# # Determine if it's a colormap or direct color
			# if color_or_cmap in self.available_cmaps:
			#	  # It's a colormap
			#	  color = self.available_cmaps[color_or_cmap](0.8)[:3]
			#	  label = f"{cloud_id} ({color_or_cmap}, {int(alpha*100)}%)"
			# else:
			#	  # It's a direct color
			#	  color = color_or_cmap
			#	  label = f"{cloud_id} ({color}, {int(alpha*100)}%)"
			color = color_or_cmap
			label = f"{cloud_id} ({color}, {int(alpha*100)}%)"
			
			patch = patches.Patch(color=color, alpha=alpha, label=label)
			legend_entries.append(patch)
			
		# Add the legend
		if legend_entries:
			self.Plot.canvas.ax.legend(handles=legend_entries, loc='upper right', fontsize='small')
			
	def closeEvent(self, event):
		"""Ran when the window is closed"""
		self.dead = True
		
	def save_fig(self, file):
		"""Saves the figure as a png file"""	
		# 1) Let the user pick a file name
		fname, _ = QFileDialog.getSaveFileName(
			self,
			"Save Plot As…",
			"",
			"PNG Files (*.png);;All Files (*)"
		)
		
		if not fname: return

		self.Plot.canvas.figure.savefig(
			fname,
			dpi=300,
			bbox_inches='tight'
		)
	