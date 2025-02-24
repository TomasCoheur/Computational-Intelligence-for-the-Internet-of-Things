import pants
import pandas as pd
import time

start_time = time.perf_counter()
ecopoints_to_visit = pd.read_csv("ListEcoPoints.csv")
filename = "Project2_DistancesMatrix.xlsx"
distance_file = pd.read_excel(filename, index_col=[0])
list_of_ecopoints = ecopoints_to_visit["EcoPoints"].values.tolist()
list_of_ecopoints.insert(0, 'C')


def get_distance(actual_ecopoint, next_ecopoint):
    return distance_file[actual_ecopoint][next_ecopoint]


world = pants.World(list_of_ecopoints, get_distance)
world.data(0)

solver = pants.Solver()
solution = solver.solve(world)

print("Distance: ", int(solution.distance))
print(solution.tour)    # Nodes visited in order
#print(solution.path)    # Edges taken in order
end_time = time.perf_counter()
print("Time calculating: ", end_time - start_time, "seconds")