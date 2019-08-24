import random
import numpy as np

class DiscreteDistributionSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution
        self.__init_table()

    def __init_table(self):
        K = len(self.distribution)
        values = [K * prob for prob in self.distribution]

        smalls = []
        larges = []

        for index, value in enumerate(values):
            if value < 1.0:
                smalls.append(index)
            elif value >= 1.0:
                larges.append(index)
        
        other_sides = [_ for _ in range(len(values))]
        while len(smalls) > 0  and len(larges) > 0:
            small = smalls.pop()
            large = larges.pop()

            other_sides[small] = large

            total = values[small] + values[large]
            left = total - 1.0
            values[large] = left
            if left < 1.0:
                smalls.append(large)
            else:
                larges.append(large)
        
        self.values = values
        self.other_sides = other_sides

    def sampling(self):
        index = np.random.randint(len(self.values))
        random_value = np.random.rand()
        if random_value < self.values[index]:
            return index
        else:
            return self.other_sides[index]

if __name__ == '__main__':
    sampler = DiscreteDistributionSampler([0.1, 0.3, 0.5, 0.1])
    data = []
    for i in range(1000000):
        data.append(sampler.sampling())

    data = np.array(data)
    for i in range(4):
        d = data[ data == i]
        print("{}: {}".format(i, d.shape[0] / data.shape[0]))


        