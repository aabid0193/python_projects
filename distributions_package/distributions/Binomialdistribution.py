import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .Generaldistribution import Distribution

class Binomial(Distribution):
    """
    Binomial distribution class for calculating and
    visualizing a binomial distribution

    Attributes:
        mean (float) - representing the mean value of the distribution
        stdev (float) - representing the standard deviation of the distribution
        data_list (list) - a list of floats extracted from the data file
        p (float) - represents the probability of an event occurring
        n (int) - represents the total number of trials
    """

    def __init__(self, prob = 0,  size = 20):

        self.p = prob
        self.n = size

        Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev())

    def calculate_mean(self):
        """
        Function to calculate the mean from p and n

        Inputs:
            None
        Return:
            self.mean (float) - mean of the dataset
        """
        self.mean = self.n * self.p

        return self.mean

    def calculate_stdev(self):
        """
        Function to calculate the standard deviation from p and n

        Inputs:
            None
        Returns:
            self.stdev (float) - standard deviation of the dataset
        """
        self.stdev = math.sqrt(self.n * self.p * (1-self.p))

        return self.stdev

    def replace_stats_with_data(self):
        """
        Function to calculate p and n from the data set

        Inputs:
            None
        Returns:
            self.p (float) - returns value for p
            self.n (float) - return value for n
        """
        self.n = len(self.data)
        self.p = 1.0 * sum(self.data) / len(self.data)
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()

        return self.p, self.n

    def plot_bar(self):
        """
        Function to output a histogram of the instance variable data
        using matplotlib pyplot library.

        Inputs:
            None
        Returns:
            None
        """
        plt.bar(self.data)
        #plt.bar(x = ['0', '1'], height = [(1 - self.p) * self.n, self.p * self.n])
        plt.title("Barplot for Binomial Distribution")
        plt.xlabel("Data")
        plt.ylabel("Count")

    def pdf(self, x):
        """
        Probability density function calculator for the binomial distribution.

        Inputs:
            x (float) -  point for calculating the probability density function
        Returns:
            binomial_pdf (float) -  probability density function output
        """
        n_choose_x = math.factorial(self.n)/(math.factorial(x) * math.factorial(self.n - x))
        probs = ((self.p)**x)*((1-self.p)**(self.n - x))

        binomial_pdf = n_choose_x * probs

        return binomial_pdf

    def plot_bar_pdf(self):
        """
        Function to plot the pdf of the binomial distribution

        Inputs:
            None
        Returns:
            x (list) - x values for the pdf plot
            y (list) - y values for the pdf plot
        """

        x = []
        y = []
        # calculate the x values to visualize
        for i in range(self.n + 1):
            x.append(i)
            y.append(self.pdf(i))
        # make the plots
        plt.bar(x, y)
        plt.title('Distribution of Outcomes')
        plt.ylabel('Probability')
        plt.xlabel('Outcome')

        plt.show()
        return x, y

    def __add__(self, other):
        """
        Function to add together two binomial distributions with equal p

        Inputs:
            other (Binomial) - another binomial instance
        Returns:
            sum (Binomial) - Output Binomial distribution after adding two Binomials
        """
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise

        result = Binomial()
        result.n = self.n + other.n
        result.p = self.p
        result.calculate_mean()
        result.calculate_stdev()

        return result

    def __repr__(self):
        """
        Function to output the characteristics of the Binomial instance

        Inputs:
            None
        Returns:
            (string) - characteristics of the Binomial
        """
        return f"mean {self.mean}, standard deviation {self.stdev}, p {self.p}, n {self.n}"
