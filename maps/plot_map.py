import matplotlib.pyplot as plt
import pandas as pd

folder_path = 'maps/snake_map_training/'

left_side_road_coordinates = pd.read_csv(f'{folder_path}/road_left.csv')
right_side_road_coordinates = pd.read_csv(f'{folder_path}/road_right.csv')
middle_side_road_coordinates = pd.read_csv(f'{folder_path}/road_middle.csv')

fig, ax = plt.subplots()

ax.plot(left_side_road_coordinates.x_values, left_side_road_coordinates.y_values, color='blue', label='left_side')
ax.plot(right_side_road_coordinates.x_values, right_side_road_coordinates.y_values, color='red', label='right_side')
ax.plot(middle_side_road_coordinates.x_values, middle_side_road_coordinates.y_values, color='orange', label='middle')

ax.set(xlabel='x', ylabel='y')
ax.legend()
plt.show()