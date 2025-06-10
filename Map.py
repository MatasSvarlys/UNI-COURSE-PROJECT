

class Map:
    def __init__(self, file_location=None):
        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                first_line = f.readline().strip()
                self.width, self.height = map(int, first_line.split())
                self.map_data = []
                for _ in range(self.height):
                    row = f.readline().strip().split()
                    self.map_data.append([int(tile) for tile in row])
        # If no file location is provided, initialize an empty map
        else:
            self.width = 0
            self.height = 0
            self.map_data = []


    def set_tile(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.map_data[y][x] = value

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map_data[y][x]
        return None

    def display(self):
        for row in self.map_data:
            print(" ".join(str(tile) for tile in row))