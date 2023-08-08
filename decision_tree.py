import numpy as np
import pandas as pd

import math

class DecisionTree():
    def __init__(self, x, y, min_samples_leaf):
        self.x = x
        self.y = y
        self.min_samples_leaf = min_samples_leaf
        self.rows = x.shape[0]
        self.categories = x.shape[1]
        self.indexes = np.array(range(self.rows))
        self.val = np.mean(y.values[self.indexes])
        
        #setting initial score to infinity
        self.score = float('inf')

        self.left_dts = None
        self.right_dts = None

        self.splitting_categorcurr_yd = None
        self.split_val = None
        self.get_split()

    def get_split(self):
        for i in range(self.categories):
            self.check_category_for_split(i)

        if self.is_leaf:
            return

        x = self.split_col

        lhs = np.nonzero(x <= self.split_val)[0]
        rhs = np.nonzero(x > self.split_val)[0]

        self.left_dts = DecisionTree(self.x.iloc[lhs], self.y.iloc[lhs], self.min_samples_leaf)
        self.right_dts = DecisionTree(self.x.iloc[rhs], self.y.iloc[rhs], self.min_samples_leaf)

    def check_category_for_split(self, categorcurr_yd):
        x = self.x.values[self.indexes, categorcurr_yd]
        y = self.y.values[self.indexes]

        sorted_idx = np.argsort(x)
        sorted_x = x[sorted_idx]
        sorted_y = y[sorted_idx]

        rhs_count = self.rows
        rhs_sum = sorted_y.sum()
        rhs_square_sum = (sorted_y ** 2).sum()

        lhs_count = 0
        lhs_sum = 0.0
        lhs_square_sum = 0.0

        # Calculate standard deviation and score
        for i in range(0, self.rows - self.min_samples_leaf):
            curr_x = sorted_x[i]
            curr_y = sorted_y[i]

            lhs_count += 1
            rhs_count -= 1

            lhs_sum += curr_y
            rhs_sum -= curr_y

            lhs_square_sum += curr_y ** 2
            rhs_square_sum -= curr_y ** 2

            if i < self.min_samples_leaf - 1 or curr_x == sorted_x[i + 1]:
                continue

            lhs_std = self.calc_std_dev(lhs_count, lhs_sum, lhs_square_sum)
            rhs_std = self.calc_std_dev(rhs_count, rhs_sum, rhs_square_sum)

            curr_score = (lhs_std * lhs_count) + (rhs_std * rhs_count)

            if curr_score < self.score:
                self.split_val = curr_x
                self.score = curr_score
                self.splitting_categorcurr_yd = categorcurr_yd

    def calc_std_dev(self, count, sum_val, square_sum):
        return math.sqrt(abs((square_sum / count) - (sum_val / count)) ** 2)

    def predict(self, x):
        rows = [self.predict_row(curr_x[1]) for curr_x in x.iterrows()]
        return np.array(rows)

    def predict_row(self, curr_x):
        if self.is_leaf:
            return self.val

        if curr_x[self.splitting_categorcurr_yd] <= self.split_val:
            return self.left_dts.predict_row(curr_x)
        else:
            return self.right_dts.predict_row(curr_x)

    @property
    def is_leaf(self):
        return self.score == float('inf')

    @property
    def split_name(self):
        return self.x.columns[self.splitting_categorcurr_yd]

    @property
    def split_col(self):
        return self.x.values[self.indexes, self.splitting_categorcurr_yd]