Make sure all three jpg files(flower.jpg, tower.jpg, building.jpg) are in the same directory with two python file(edge.py, corner.py)
Notice: during running, the program might generate divide by 0 warning but this will not cause program to crush

Before running the program make sure your python3 has the following package installed:
skimage
scipy
pylab
matplotlib
numpy

Edge detector:
in command line type: python3 edge.py
the program will ask you to give input file name, and threshold value.

For flower_edge:
type in "flower.jpg"
then 0.003 for high threshold, 0.001 for low threshold

For tower_edge:
type in "tower.jpg"
then 0.01 for high threshold, 0.005 for low threshold

Corner detector
in command line type: python3 edge.py
the program will ask you to give input file name, and threshold value.

For flower_corner:
type in "flower.jpg"
then 0.02 for the threshold value

For building_corner:
type in "building.jpg"
then 0.1 for the threshold value
