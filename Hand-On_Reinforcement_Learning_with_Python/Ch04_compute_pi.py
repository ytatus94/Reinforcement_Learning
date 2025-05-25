import numpy as np
import math
import random
import matplotlib.pyplot as plt
%matplotlib inline

square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 1000
arc = np.linspace(0, np.pi/2, 100) # 把 0 到 90 度 (pi/2) 分成 100 個點

def generate_points(size):
    x = random.random() * size
    y = random.random() * size
    return (x, y)

# 判斷是否位於圓內
def is_in_circle(point, size):
    return math.sqrt(point[0]**2 + point[1]**2) <= size

# 利用圓的面積與正方形面積的比例等於圓內點數與正方形內點數的比例來計算 pi
# pi * r^2 / a^2 = 圓內點數 / 正方形內點數
# 假設 r = a/2 會得到 pi = 4 * (圓內點數 / 正方形內點數)
def compute_pi(points_inside_circle, points_inside_square):
    return 4 * (points_inside_circle / points_inside_square)

plt.axes().set_aspect("equal")
plt.plot(1 * np.cos(arc), 1 * np.sin(arc)) # 半徑是 r, 則 x = r cos\theta, y = r sin\theta

for i in range(sample_size):
    point = generate_points(square_size)

    plt.plot(point[0], point[1], "c.")
    points_inside_square += 1

    if is_in_circle(point, square_size):
        points_inside_circle += 1

print("Approximate value of pi is {}".format(compute_pi(points_inside_circle, points_inside_square)))
