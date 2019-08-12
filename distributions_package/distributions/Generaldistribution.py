import math
import numpy as np

class Distribution:
    #contains general distribution parent class
    def __init__(self, mu = 0, sigma = 1):
        """
        General distribution class for calculating and visualizing
        probability distributions.

        Attributes:
            mean (float) - Average value representation for the distribution
            stdev (float) - Standard Deviation value for the distribution
            data_list (list) - list of floats extracted from the data file
        """
        self.mean = mu
        self.stdev = sigma
        self.data = []

    def read_data_file(self, file_name):
        """
        Read data from txt file.

        Inputs:
            file_name (string) - name of txt file to read
        Returns:
            None
        """
        with open(file_name) as f:
            data_list = []
            line = f.readline()
            while line:
                data_list.append(int(line))
                line = f.readline()
        file.close()

        self.data = data_list
