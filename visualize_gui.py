import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QGridLayout
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QFileDialog
from PyQt6 import QtGui, QtCore
import open3d as o3d
import win32gui
import numpy as np
import matplotlib.pyplot as plt
import os

class PointCloudViewer(QMainWindow):
  def __init__(self, cloud_name):
    super().__init__()
    self.setWindowTitle("Point Cloud Viewer")

    self.widget = QWidget(self)
    self.setCentralWidget(self.widget)
    self.layout = QGridLayout()
    self.widget.setLayout(self.layout)

    self.right_widget = QWidget(self)
    self.layout.addWidget(self.right_widget, 0, 1, 2, 1)
    self.right_layout = QVBoxLayout()
    self.right_widget.setLayout(self.right_layout)

    self.cloud_name_btn = QPushButton(cloud_name)
    self.cloud_name_btn.clicked.connect(self.open_file_dialog)
    self.right_layout.addWidget(self.cloud_name_btn)

    self.export_btn = QPushButton("Export")
    self.export_btn.clicked.connect(self.export)
    self.right_layout.addWidget(self.export_btn)

    self.cloud_name = self.cloud_name_btn.text()
    # check if the file is a mesh
    if self.cloud_name.endswith(".ply") or self.cloud_name.endswith(".pcd") or self.cloud_name.endswith(".xyz"):
        self.cloud = o3d.io.read_point_cloud(self.cloud_name)
        self.points = np.asarray(self.cloud.points).copy()
        self.colors = np.asarray(self.cloud.colors).copy()
    else:
        self.cloud = o3d.io.read_triangle_mesh(self.cloud_name)
        self.points = np.asarray(self.cloud.vertices).copy()
        self.colors = np.ones((len(self.points), 3))


    self.cmap = plt.get_cmap("inferno")
    # use cmap to get the colors
    self.colors = self.cmap(self.colors[:, 0])[:, :3]

    if isinstance(self.cloud, o3d.geometry.TriangleMesh):
        self.cloud.compute_vertex_normals()
        self.cloud.compute_triangle_normals()
    else:
        self.cloud.colors = o3d.utility.Vector3dVector(self.colors)
        

    # certainties are stored in the Red channel
    self.certainties = self.colors[:, 0]
    # normalize the certainties
    self.certainties = (self.certainties - np.min(self.certainties)) / (np.max(self.certainties) - np.min(self.certainties))

    
    

    self.viewer = o3d.visualization.Visualizer()
    self.viewer.create_window()
    self.viewer.add_geometry(self.cloud)

    hwnd = win32gui.FindWindowEx(0,0,None, "Open3D")
    self.window = QtGui.QWindow.fromWinId(hwnd)
    self.windowContainer = self.createWindowContainer(self.window, self)
    self.layout.addWidget(self.windowContainer, 0, 0)

    # Outlier removal button
    outlier_widget = QWidget(self)
    outlier_layout = QGridLayout()
    outlier_widget.setLayout(outlier_layout)
    self.right_layout.addWidget(outlier_widget)

    outlier_layout.addWidget(QLabel("Statistical Outlier Removal"), 0, 0, 1, 2)
    outlier_highlight = QPushButton("Highlight Outliers")
    outlier_layout.addWidget(outlier_highlight, 1, 0, 1, 1)
    outlier_highlight.clicked.connect(self.highlight_outliers)
    outlier_remove = QPushButton("Remove Outliers")
    outlier_layout.addWidget(outlier_remove, 1, 1, 1, 1)
    outlier_remove.clicked.connect(self.remove_outliers)
    self.outlier_val = QSlider(QtCore.Qt.Orientation.Horizontal)
    self.outlier_val.setRange(0, 20)
    self.outlier_val.setValue(5)
    outlier_layout.addWidget(self.outlier_val, 3, 0, 1, 2)

    estimate_normals = QPushButton("Estimate Normals")
    estimate_normals.clicked.connect(self.estimate_normals)
    self.right_layout.addWidget(estimate_normals)

    estimate_mesh = QPushButton("Estimate Mesh")
    estimate_mesh.clicked.connect(self.estimate_mesh)
    self.right_layout.addWidget(estimate_mesh)



    reset_btn = QPushButton("Reset")
    reset_btn.clicked.connect(self.reset)
    self.right_layout.addWidget(reset_btn)

    timer = QtCore.QTimer(self)
    timer.timeout.connect(self.update_vis)
    timer.start(1)


    # add slider
    self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    self.slider.setRange(0, 100)
    self.slider.setValue(0)
    self.slider.setFixedHeight(20)

    slider_style = """
    QSlider::handle:horizontal {
        background: white;
        border: 1px solid #000000;
        width: 10px;
        height: 18px;
        margin: -5px 2;
    }

    QSlider::groove:horizontal {
        border: 1px solid #000000;
        height: 13px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
    """
    cmap_colors = self.cmap(np.linspace(0, 1, 10))[:, :3] * 255

    for i, color in enumerate(cmap_colors):
        slider_style += f"stop:{i/10} rgba({int(color[0])}, {int(color[1])}, {int(color[2])}, 255),"

    slider_style = slider_style[:-1] + ");}"

    self.slider.setStyleSheet(slider_style)

    self.slider.valueChanged.connect(self.update_point_count)
    self.layout.addWidget(self.slider, 1, 0)



    self.showMaximized()

  def open_file_dialog(self):
      file = QFileDialog.getOpenFileName(self, "Open Point Cloud", "", "Point Cloud Files (*.ply *.pcd *.xyz *.obj *.stl)")
      file = file[0]
      if True:# file:
          # get relative path to annotool folder 
          relative_path = os.path.relpath(file, os.getcwd())
          # update button text
          self.cloud_name_btn.setText(relative_path)
      self.reset()

  def export(self):
        file = QFileDialog.getSaveFileName(self, "Save Point Cloud", "", "Point Cloud Files (*.ply *.pcd *.xyz *.obj *.stl)")
        file = file[0]
        if file:
            if not file.endswith(".ply") and not file.endswith(".pcd") and not file.endswith(".xyz") and not file.endswith(".obj") and not file.endswith(".stl"):
                file += ".ply"
            #check if its a mesh
            if isinstance(self.cloud, o3d.geometry.TriangleMesh):
                o3d.io.write_triangle_mesh(file, self.cloud)
            else:
                o3d.io.write_point_cloud(file, self.cloud)



  def update_point_count(self):

      # using the certainties, we threshold the points
      threshold = self.slider.value() / 100
      certainties = self.certainties.copy()
      certainties[certainties < threshold] = 0
      certainties[certainties >= threshold] = 1


      points = self.points * certainties[:, None]
      colors = self.colors * certainties[:, None]
      # update the points
      self.cloud.points = (o3d.utility.Vector3dVector(points))
      self.cloud.colors = (o3d.utility.Vector3dVector(colors))

      self.viewer.update_geometry(self.cloud)
      self.update_vis()
      

  def highlight_outliers(self):
      # mouse loading symbol
      self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

      std = self.outlier_val.value() / 10

      # remove the outliers
      _, ind = self.cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=std)

      # display the outliers in red
      colors = self.colors.copy()
      # get the indices of the colors
      indices = np.arange(len(colors))
      # subtract the indices of the outliers
      indices = np.setdiff1d(indices, ind)
      # set the colors of the outliers to red
      colors[indices] = [1, 0, 0]

      self.cloud.colors = o3d.utility.Vector3dVector(colors)
      self.viewer.update_geometry(self.cloud)

      # reset the cursor
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

  def remove_outliers(self):
      # mouse loading symbol
      self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

      std = self.outlier_val.value() / 10

      # remove the outliers
      _, ind = self.cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=std)

      self.points = self.points[ind]
      self.colors = self.colors[ind]

      self.cloud.colors = o3d.utility.Vector3dVector(self.colors)
      self.cloud.points = o3d.utility.Vector3dVector(self.points)
      self.viewer.update_geometry(self.cloud)
      
      # reset the cursor
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

  def estimate_normals(self):
      # set the original normals to facing out from the centerpoint
      centerx = np.mean(self.points[:, 0])
      centery = np.mean(self.points[:, 1])
      centerz = np.mean(self.points[:, 2])

      self.cloud.normals = o3d.utility.Vector3dVector(self.points - np.array([centerx, centery, centerz]))
      self.cloud.normals = o3d.utility.Vector3dVector(np.asarray(self.cloud.normals) / np.linalg.norm(np.asarray(self.cloud.normals), axis=1)[:, None])
      # normalize the normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.asarray(self.cloud.normals) / np.linalg.norm(np.asarray(self.cloud.normals), axis=1)[:, None])


      self.cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30))
      self.viewer.update_geometry(self.cloud)
      self.viewer.get_render_option().point_show_normal = True
      self.update_vis()

  def estimate_mesh(self):
      # mouse loading symbol
    self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.cloud, depth=9)

    #mesh = mesh.filter_smooth_laplacian(10)

    self.viewer.clear_geometries()
    self.viewer.add_geometry(mesh)
    self.viewer.update_geometry()
    self.viewer.get_render_option().point_show_normal = False
    self.update_vis()
    # reset the cursor
    self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

  def reset(self):
      self.cloud_name = self.cloud_name_btn.text()
      self.cloud = o3d.io.read_point_cloud(self.cloud_name)
      self.colors = np.asarray(self.cloud.colors).copy()
      self.points = np.asarray(self.cloud.points).copy()
      self.certainties = self.colors[:, 0]
      self.certainties = (self.certainties - np.min(self.certainties)) / (np.max(self.certainties) - np.min(self.certainties))
      self.colors = self.cmap(self.colors[:, 0])[:, :3]
      self.cloud.colors = o3d.utility.Vector3dVector(self.colors)
      #self.cloud.normals = None
      self.slider.setValue(0)
      self.viewer.clear_geometries()
      self.viewer.add_geometry(self.cloud)
      self.viewer.update_geometry()
      self.viewer.get_render_option().point_show_normal = False
      self.update_vis()    

  def update_vis(self):
      #self.viewer.update_geometry()
      self.viewer.poll_events()
      self.viewer.update_renderer()
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)







if __name__ == "__main__":
  app = QApplication(sys.argv)

  # Create the GUI window
  window = PointCloudViewer("big_bunny.obj")#("point_cloud_1080_bam.ply")
  window.show()
  sys.exit(app.exec())