import json


class NavigationConfig:
    def __init__(self, config_file):
        with open(config_file, "r") as json_file:
            self.config_json = json.load(json_file)

            self.mode = self.find_mode()
            self.key_points = self.find_key_points()
            self.test_route_points = self.find_test_route_points()

    def find_mode(self):
        mode = self.config_json['mode']

        if mode == 'test' or mode == 'drone':
            return mode
        else:
            raise Exception("Unknown mode!")

    def find_key_points(self):
        points = []
        for p in self.config_json['key_points']:
            points.append((p['x'], p['y']))

        return points

    def find_test_route_points(self):
        if self.mode == 'test':
            points = []
            for p in self.config_json['test_route_points']:
                points.append((p['x'], p['y']))

            return points
        else:
            return []
