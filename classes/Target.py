import numpy as np
from shapely.geometry import Polygon, Point



class Target_info():
    def __init__(self, all_builds, test_name):
        self.num_targets = np.random.randint(low=200, high=250)
        self.builds = all_builds
        self.generate_targets(test_name)

    def generate_targets(self, test_name):
        self.all_coords = []
        num_builds = len(self.builds)
        vals = np.random.uniform(size=(num_builds,1))
        props = vals / sum(vals)
        min_height = 0.0

        for i in range(num_builds):
            build_poly = Polygon(self.builds[i].poly2D_coords)
            num = int(props[i]*self.num_targets)
            while len(self.plants[i].target_coords) < num:
                c = np.random.rand(1,2)
                target_coord = Point(c[0])
                if target_coord.within(build_poly):
                    height = np.random.uniform(low=min_height, high=self.builds[i].height)
                    coord = [c[0][0], c[0][1], height]
                    self.builds[i].target_coords.append(coord)
                    self.all_coords.append(coord)
    
    def get_all_coords(self):
        return self.all_coords