import numpy as np


class DataProcessor:
   # YOUR CODE HERE
    def __init__(self, data):
        self.data = data
        self.processed_data_ = None

    def process(self):
        if self.data:
            self.processed_data_ = self.data - np.mean(self.data)
    
    def save_to_file(self, path: str):
        if self.processed_data_ is not None:
            with open(path, 'w') as f:
                for value in self.processed_data_:
                    f.write(str(value) + '\n')
        else:
            pass

# Example of usage
processor = DataProcessor(data=[1, 2, 3, 4, 5])
processor.process()
processor.save_to_file("processed_data.txt")