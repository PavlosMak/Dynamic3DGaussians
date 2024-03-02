import torch


class VoxelGrid3D:
    """
    A wrapper around pytorch to make some common voxel grid operation higher level
    """

    def __init__(self, min_point, max_point, voxel_size):
        self.voxel_size = voxel_size
        self.half_size = self.voxel_size / 2
        self.min_point = min_point
        self.max_point = max_point
        self.data = dict()
        self.is_dense = False
        self.occupancies = None
        self.voxel_count = 0
        self.x_resolution = -1
        self.y_resolution = -1
        self.z_resolution = -1

    def make_dense(self):
        x_bins, y_bins, z_bins = (self.max_point - self.min_point) / self.voxel_size
        x_bins, y_bins, z_bins = int(x_bins), int(y_bins), int(z_bins)
        self.occupancies = torch.zeros((x_bins, y_bins, z_bins))
        self.voxel_count = x_bins * y_bins * z_bins
        self.is_dense = True
        self.x_resolution = x_bins
        self.y_resolution = y_bins
        self.z_resolution = z_bins

    def add_to_occupancy(self, point, value):
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        i, j, k = int(i), int(j), int(k)
        self.occupancies[i][j][k] += value

    def get_occupancy(self, point):
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        i, j, k = int(i), int(j), int(k)
        return self.occupancies[i][j][k]

    def add_data(self, point, data):
        assert len(point) == 3, "Expected point to be 3D"
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        key = (int(i), int(j), int(k))
        if key not in self.data:
            self.data[key] = []
            self.voxel_count += 1
        self.data[key].append(data)

    def get_data(self, point):
        """Returns empty list if no data on the point's voxel"""
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        key = (int(i), int(j), int(k))
        if key not in self.data:
            return []
        return self.data[key]

    def get_voxel_center_point(self, i: int, j: int, k: int):
        x = self.min_point[0] + i * self.voxel_size + self.half_size
        y = self.min_point[1] + j * self.voxel_size + self.half_size
        z = self.min_point[2] + k * self.voxel_size + self.half_size
        return torch.tensor([x, y, z])
