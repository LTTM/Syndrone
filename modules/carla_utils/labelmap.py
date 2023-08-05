import numpy as np


labels = [
    "Buildings",
    "Fences",
    "Other",
    "Poles",
    "RoadLines",
    "Roads",
    "Sidewalks",
    "Vegetation",
    "Walls",
    "TrafficSigns",
    "Sky",
    "Ground",
    "Bridge",
    "RailTrack",
    "GuardRail",
    "TrafficLight",
    "Static",
    "Dynamic",
    "Water",
    "Terrain",
    "Persons",
    "Riders",
    "Cars",
    "Trucks",
    "Busses",
    "Trains",
    "Motorcycles",
    "Bycicles"
]

bblabels = {
    #"Poles": 5,
    #"TrafficSigns": 12,
    #"TrafficLight": 18,
    #"Static": 19,
    #"Dynamic": 20,
    "Persons": 40,
    "Riders": 41,
    "Cars": 100,
    "Trucks": 101,
    "Busses": 102,
    "Trains": 103,
    "Motorcycles": 104,
    "Bycicles": 105
}


cmap = np.zeros((256,3), dtype=np.uint8)
cmap[  1] = [ 70, 70, 70] # building
cmap[  2] = [190,153,153] # fence
cmap[  3] = [180,220,135] # other
cmap[  5] = [153,153,153] # pole
cmap[  6] = [255,255,255] # road line
cmap[  7] = [128, 64,128] # road
cmap[  8] = [244, 35,232] # sidewalk
cmap[  9] = [107,142, 35] # vegetation
cmap[ 11] = [102,102,156] # wall
cmap[ 12] = [220,220,  0] # traffic sign
cmap[ 13] = [ 70,130,180] # sky
cmap[ 14] = [ 81,  0, 81] # ground
cmap[ 15] = [150,100,100] # bridge
cmap[ 16] = [230,150,140] # rail track
cmap[ 17] = [180,165,180] # guard rail
cmap[ 18] = [250,170, 30] # traffic light
cmap[ 19] = [110,190,160] # static
cmap[ 20] = [111, 74,  0] # dynamic
cmap[ 21] = [ 45, 60,150] # water
cmap[ 22] = [152,251,152] # terrain
cmap[ 40] = [220, 20, 60] # person
cmap[ 41] = [255,  0,  0] # rider
cmap[100] = [  0,  0,142] # car
cmap[101] = [  0,  0, 70] # truck
cmap[102] = [  0, 60,100] # bus
cmap[103] = [  0, 80,100] # train
cmap[104] = [  0,  0,230] # motorcycle
cmap[105] = [119, 11, 32] # bicycle
