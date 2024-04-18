import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QGridLayout
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QFileDialog, QButtonGroup, QSpinBox
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
    # show backsides of the triangles
    #self.viewer.get_render_option().mesh_show_back_face = True

    hwnd = win32gui.FindWindowEx(0,0,None, "Open3D")
    self.window = QtGui.QWindow.fromWinId(hwnd)
    self.windowContainer = self.createWindowContainer(self.window, self)
    self.layout.addWidget(self.windowContainer, 0, 0)

    # load in the obj to compare to
    self.obj = o3d.io.read_triangle_mesh("big_bunny.obj")
    # scale the obj to the same size as the point cloud
    # get the minimum and maximum y values of the point cloud
    min_y = 1080 - 1047
    max_y = 1080 - 59
    # get the minimum and maximum y values of the obj   
    min_y_obj = np.min(np.asarray(self.obj.vertices)[:, 1])
    max_y_obj = np.max(np.asarray(self.obj.vertices)[:, 1])
    # scale the obj to the same size as the point cloud
    self.obj.scale((max_y - min_y) / (max_y_obj - min_y_obj), center=self.obj.get_center())
    
    # the center of the point cloud lies at half of the maximum y value
    self.obj.translate((15, 418, 7))

    self.obj.compute_vertex_normals()
    self.obj.compute_triangle_normals()

    # display image of the origin
    # image name is in the same directory as the point cloud, the file 0001.png
    img_name = os.path.join(os.path.dirname(self.cloud_name), "0001.png")

    self.origin_img = QLabel()
    self.origin_img.setPixmap(QtGui.QPixmap(img_name).scaledToWidth(350))

    self.right_layout.addWidget(self.origin_img)

    self.calculate_distances()

    # radio button for either certainties or distances
    self.cd_btn_group = QButtonGroup()
    self.cd_btn_group.setExclusive(True)
    self.certainties_btn = QCheckBox("Certainties")
    self.certainties_btn.stateChanged.connect(self.update_point_count)
    self.distances_btn = QCheckBox("Distances")
    #self.distances_btn.stateChanged.connect(self.update_point_count)
    self.cd_btn_group.addButton(self.certainties_btn)
    self.cd_btn_group.addButton(self.distances_btn)
    self.certainties_btn.setChecked(True)
    self.right_layout.addWidget(self.certainties_btn)
    self.right_layout.addWidget(self.distances_btn)

    self.max_distance = QLabel(f"Max Distance: {np.max(self.distances):.2f}")
    self.right_layout.addWidget(self.max_distance)
    self.min_distance = QLabel(f"Min Distance: {np.min(self.distances):.2f}")
    self.right_layout.addWidget(self.min_distance)
    self.mean_error = QLabel(f"Mean Distance: {np.mean(self.distances):.2f}")
    self.right_layout.addWidget(self.mean_error)
    self.point_count = QLabel(f"Point Count: {len(self.points)}")   
    self.right_layout.addWidget(self.point_count)

    self.show_origin = QCheckBox("Show Origin")
    self.show_origin.stateChanged.connect(self.show_origin_obj)
    self.right_layout.addWidget(self.show_origin)

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

  def show_origin_obj(self):
    if self.show_origin.isChecked():
        self.viewer.add_geometry(self.obj)
    else:
        self.viewer.remove_geometry(self.obj)
    self.update_vis()


  def calculate_distances(self):
        # mouse loading symbol
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

        # calculates the distances from each point to the nearest point in the obj
        self.distances = np.zeros(len(self.points))
        kd_tree = o3d.geometry.KDTreeFlann(self.obj)

        for i, point in enumerate(self.points):
            [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
            self.distances[i] = np.linalg.norm(point - np.asarray(self.obj.vertices)[idx[0]])

        

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
      

      self.max_distance.setText(f"Max Distance: {np.max(self.distances[certainties == 1]):.2f}")
      self.mean_error.setText(f"Mean Distance: {np.mean(self.distances[certainties == 1]):.2f}")
      self.min_distance.setText(f"Min Distance: {np.min(self.distances[certainties == 1]):.2f}")
      self.point_count.setText(f"Point Count: {int(np.sum(certainties))}")

      points = self.points * certainties[:, None]
      if self.certainties_btn.isChecked():
          colors = self.colors * certainties[:, None]
      else:
          distances = self.distances * certainties
          # normalize the distances
          distances = 1 - ((distances - np.min(distances)) / (np.max(distances) - np.min(distances)))
          colors = self.cmap(distances)[:, :3]


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
      self.certainties = self.certainties[ind]
      self.distances = self.distances[ind]

      self.cloud.colors = o3d.utility.Vector3dVector(self.colors)
      self.cloud.points = o3d.utility.Vector3dVector(self.points)
      self.viewer.update_geometry(self.cloud)
      
      # reset the cursor
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

  def estimate_normals(self):
      # mouse loading symbol
      self.setCursor(QtCore.Qt.CursorShape.WaitCursor)

      # estimate normals based upon the weight of the neighbouring points, pointing in the other direction
      
      # go through the points and get the neighbouring points

      # create empty normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.zeros_like(self.points))

      # create a KDTree
      kd_tree = o3d.geometry.KDTreeFlann(self.cloud)
      # get the indices of the nearest neighbours
      for i, point in enumerate(self.points):
            
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, 200)


        nn_points = np.asarray(self.points)[idx]
        # check if nan is in the points
        if np.isnan(nn_points).any():
            print("Nan in points")
            continue

        nn_dists = np.linalg.norm(nn_points - point, axis=1)
        # normalize the distances
        nn_dists = 1 - ((nn_dists - np.min(nn_dists)) / (np.max(nn_dists) - np.min(nn_dists)))

        

        # create a weighted average
        avg = np.average(nn_points, axis=0, weights=nn_dists)

        # set the normal to the difference between the point and the average
        inverse_normal = point - avg
        # normalize the normal
        inverse_normal = inverse_normal / np.linalg.norm(inverse_normal)
        # inverse the normal
        normal = inverse_normal

        self.cloud.normals[i] = normal


      # normalize the normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.asarray(self.cloud.normals) / np.linalg.norm(np.asarray(self.cloud.normals), axis=1)[:, None])


      # set the original normals to facing out from the centerpoint
      #centerx = np.mean(self.points[:, 0])
      #centery = np.mean(self.points[:, 1])
      #centerz = np.mean(self.points[:, 2])
#
      #self.cloud.normals = o3d.utility.Vector3dVector(self.points - np.array([centerx, centery, centerz]))
      #self.cloud.normals = o3d.utility.Vector3dVector(np.asarray(self.cloud.normals) / np.linalg.norm(np.asarray(self.cloud.normals), axis=1)[:, None])
      ## normalize the normals
      #self.cloud.normals = o3d.utility.Vector3dVector(np.asarray(self.cloud.normals) / np.linalg.norm(np.asarray(self.cloud.normals), axis=1)[:, None])
#
#
      self.cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30))
      self.viewer.update_geometry(self.cloud)
      self.viewer.get_render_option().point_show_normal = True
      self.update_vis()

      # reset the cursor
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

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

      # set image to origin
      img_name = os.path.join(os.path.dirname(self.cloud_name), "0001.png")
      self.origin_img.setPixmap(QtGui.QPixmap(img_name).scaledToWidth(350))



      self.cloud = o3d.io.read_point_cloud(self.cloud_name)
      self.colors = np.asarray(self.cloud.colors).copy()
      self.points = np.asarray(self.cloud.points).copy()
      self.certainties = self.colors[:, 0]
      self.certainties = (self.certainties - np.min(self.certainties)) / (np.max(self.certainties) - np.min(self.certainties))
      self.colors = self.cmap(self.colors[:, 0])[:, :3]
      self.cloud.colors = o3d.utility.Vector3dVector(self.colors)
      self.calculate_distances()
      self.slider.setValue(0)
      self.update_point_count()
      #self.cloud.normals = None
      self.viewer.clear_geometries()
      self.viewer.add_geometry(self.cloud)
      #self.viewer.add_geometry(self.obj)
      self.viewer.update_geometry()
      self.viewer.get_render_option().point_show_normal = False
      self.show_origin.setChecked(False)
      self.update_vis()    
      
  def update_vis(self):
      #self.viewer.update_geometry()
      self.viewer.poll_events()
      self.viewer.update_renderer()
      self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)







if __name__ == "__main__":
  app = QApplication(sys.argv)

  # Create the GUI window
  window = PointCloudViewer("..\scratch\marble_100mm_10mm\point_cloud_marble_100mm_10mm.ply")
  window.show()
  sys.exit(app.exec())