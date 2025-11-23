from typing import Optional

import numpy as np
import open3d as o3d

from demo.types import PointCloudFrame


class PointCloudView:
    def __init__(self, window_name: str = "PointCloud View") -> None:
        self.window_name = window_name
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.initialized = False

    def _init_vis(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.initialized = True

    def render(self, pcd_frame: PointCloudFrame):
        if not self.initialized:
            self._init_vis()
        assert self.pcd is not None and self.vis is not None
        self.pcd.points = o3d.utility.Vector3dVector(pcd_frame.points.astype(np.float64))
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
