import open3d as o3d
import numpy as np
 
print("Load a ply point cloud, print it, and render it")
# ply_point_cloud = o3d.data.PLYPointCloud()
plyname = './output/'+'sam_v4_refit_stage1_v4'+'/log/pred_22001_0.ply'
#读点云
pcd = o3d.io.read_point_cloud(plyname)
print(pcd)
print(np.asarray(pcd.points))
#点云显示
o3d.visualization.draw_geometries([pcd])

