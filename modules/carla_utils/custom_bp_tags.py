import time
import carla
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# .5 <= bbox volume <= 3
cars =   [
            'vehicle.audi.a2',
            'vehicle.nissan.micra',
            'vehicle.audi.tt',
            'vehicle.mercedes.coupe_2020',
            'vehicle.bmw.grandtourer',
            'vehicle.micro.microlino',
            'vehicle.ford.mustang',
            'vehicle.chevrolet.impala',
            'vehicle.lincoln.mkz_2020',
            'vehicle.citroen.c3',
            'vehicle.dodge.charger_police',
            'vehicle.nissan.patrol',
            'vehicle.jeep.wrangler_rubicon',
            'vehicle.mini.cooper_s',
            'vehicle.mercedes.coupe',
            'vehicle.dodge.charger_2020',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.tesla.model3',
            'vehicle.audi.etron',
            'vehicle.lincoln.mkz_2017',
            'vehicle.dodge.charger_police_2020',
            'vehicle.mini.cooper_s_2021',
            'vehicle.nissan.patrol_2021'
         ]
# bbox volume > 3.1
trucks = [
            'vehicle.ford.ambulance',
            'vehicle.carlamotors.firetruck',
            'vehicle.carlamotors.carlacola',
            'vehicle.tesla.cybertruck',
            'vehicle.mercedes.sprinter',
            'vehicle.volkswagen.t2',
            'vehicle.lttm.truck02_c11',
            'vehicle.lttm.truck02_c12',
            'vehicle.lttm.truck02_c13',
            'vehicle.lttm.truck02_c21',
            'vehicle.lttm.truck02_c22',
            'vehicle.lttm.truck02_c23',
            'vehicle.lttm.truck02_c31',
            'vehicle.lttm.truck02_c32',
            'vehicle.lttm.truck02_c33',
            'vehicle.lttm.truck02_c41',
            'vehicle.lttm.truck02_c42',
            'vehicle.lttm.truck02_c43',
            'vehicle.lttm.truck02_c51',
            'vehicle.lttm.truck02_c52',
            'vehicle.lttm.truck02_c53',
            'vehicle.lttm.truck04_c1',
            'vehicle.lttm.truck04_c2',
            'vehicle.lttm.truck04_c3',
            'vehicle.lttm.truck04_c4',
            'vehicle.lttm.truck04_c5',
            'vehicle.lttm.truck04_c6',
            'vehicle.lttm.truck04_c7',
            'vehicle.lttm.truck04_c8',
            'vehicle.lttm.truck04_c9',
         ]
# added
busses = [
            'vehicle.lttm.bus01_c1',
            'vehicle.lttm.bus01_c2',
            'vehicle.lttm.bus01_c3',
            'vehicle.lttm.bus01_c4',
            'vehicle.lttm.bus01_c5',
            'vehicle.lttm.bus01_c6',
            'vehicle.lttm.bus01_c7',
            'vehicle.lttm.bus01_c8',
            'vehicle.lttm.bus02_c11',
            'vehicle.lttm.bus02_c12',
            'vehicle.lttm.bus02_c13',
            'vehicle.lttm.bus02_c21',
            'vehicle.lttm.bus02_c22',
            'vehicle.lttm.bus02_c23',
            'vehicle.lttm.bus02_c31',
            'vehicle.lttm.bus02_c32',
            'vehicle.lttm.bus02_c33'
         ]
# added
trains = [
            'vehicle.lttm.train01',
            'vehicle.lttm.train02'
         ]
# bbox volume < .5
mbikes = [
            'vehicle.harley-davidson.low_rider',
            'vehicle.yamaha.yzf',
            'vehicle.kawasaki.ninja',
            'vehicle.vespa.zx125'
          ]
# bbox volume < .5
bikes =  [
            'vehicle.bh.crossbike',
            'vehicle.gazelle.omafiets',
            'vehicle.diamondback.century'
         ]


two_wheel = mbikes+bikes


def custom_tag(bp):
    # extract blueprint id (string)
    bp_id = bp.id
    
    # the only blueprints that require attention 
    # are the vehicle and walker ones
    if bp_id.startswith('walker'):
        return {4: 40} # pedestrian->person
    if bp_id in cars:
        return {4:41, 10:100} # pedestrian->rider, vehicle->car
    if bp_id in trucks:
        return {0:101, 4:41, 10:101} # pedestrian->rider, vehicle->truck
    if bp_id in busses:
        return {0:102, 4:41, 10:102} # pedestrian->rider, vehicle->bus
    if bp_id in trains:
        return {0:103, 4:41, 10:103} # pedestrian->rider, vehicle->train
    if bp_id in mbikes:
        return {4:41, 10:104} # pedestrian->rider, vehicle->motorcycles
    if bp_id in bikes:
        return {4:41, 10:105} # pedestrian->rider, vehicle->bicycle

    return None


def override_parked_vehicles(world, blueprint_library, verbose=True):
    parked_objs = world.get_environment_objects(carla.CityObjectLabel.Vehicles)  # Bycicles
    trs = [carla.Transform(obj.bounding_box.location, obj.bounding_box.rotation) for obj in parked_objs]
    types = [get_bp_list_from_bbox_volume(get_bbox_volume(obj.bounding_box)) for obj in parked_objs]
    heights = [obj.bounding_box.extent.z for obj in parked_objs]
    spawned = []
    spawned_v = []
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # blueprint_library = world.get_blueprint_library()
    if verbose:
        pbar = tqdm(zip(trs, types, heights), total=len(trs), desc="Spawning parked vehicles...", leave=False)
    else:
        pbar = zip(trs, types, heights)

    for tr, bp_list, height in pbar:
        bp = blueprint_library.filter(random.choice(bp_list))[0]
        tr.location.z -= height
        tr.location.z = tr.location.z if tr.location.z > 0.27 else 0.27
        
        if len(spawned) > 0:
            diff = ((np.array(spawned)-np.array([[tr.location.x, tr.location.y, tr.location.z]]))**2).sum(axis=1)
            if np.all(diff > .1):
                v = spawn(world, bp, tr)
                spawned.append([tr.location.x, tr.location.y, tr.location.z])
        else:
            v = spawn(world, bp, tr)
            spawned.append([tr.location.x, tr.location.y, tr.location.z])
        spawned_v.append(v)
    print("Done!")
    return spawned_v


def spawn_static_vehicles(world, map_name, blueprint_library):
    train_tfs = []
    tram_tfs = []
    if "town01" in map_name.lower():
        train_tfs = [carla.Transform(carla.Location(x=227.362946, y=170.501678, z=0.35), carla.Rotation(pitch=0, yaw=90, roll=0.0))]
        tram_tfs = [carla.Transform(carla.Location(x=88.347214, y=257.318848, z=0.5), carla.Rotation(pitch=0, yaw=90, roll=0.0)),
                    carla.Transform(carla.Location(x=149, y=134, z=0.5), carla.Rotation(pitch=0, yaw=0, roll=0.0)),
                    carla.Transform(carla.Location(x=217, y=60, z=0.5), carla.Rotation(pitch=0, yaw=0, roll=0.0))]
    elif "town02" in map_name.lower():
        train_tfs = [carla.Transform(carla.Location(x=168.530640, y=141.598047, z=0.2), carla.Rotation(pitch=0, yaw=175, roll=0.0))]
        tram_tfs = [carla.Transform(carla.Location(x=124.939301, y=301.929443, z=0.5), carla.Rotation(pitch=0.0, yaw=180, roll=0.0)),
                    carla.Transform(carla.Location(x=-8., y=223, z=0.5), carla.Rotation(pitch=0.0, yaw=90, roll=0.0))]
    elif "town03" in map_name.lower():
        train_tfs = [carla.Transform(carla.Location(x=135.107071, y=228.658752, z=13.139), carla.Rotation(pitch=0, yaw=-180, roll=0.0)),
                        carla.Transform(carla.Location(x=0, y=228.658752, z=13.139), carla.Rotation(pitch=0, yaw=-180, roll=0.0)),
                        carla.Transform(carla.Location(x=135.107071, y=-223.658752, z=12.139), carla.Rotation(pitch=0, yaw=-180, roll=0.0)),
                        carla.Transform(carla.Location(x=0, y=-223.658752, z=13), carla.Rotation(pitch=0, yaw=-180, roll=0.0)),
                        carla.Transform(carla.Location(y=135.107071, x=268.658752, z=11.239), carla.Rotation(pitch=0, yaw=-90, roll=0.0)),
                        carla.Transform(carla.Location(y=0, x=268.658752, z=11.239), carla.Rotation(pitch=0, yaw=-90, roll=0.0)),
                        carla.Transform(carla.Location(y=120.107071, x=-140.658752, z=13.139), carla.Rotation(pitch=0, yaw=-90, roll=0.0)),
                        carla.Transform(carla.Location(y=0, x=-140.658752, z=13), carla.Rotation(pitch=0, yaw=-90, roll=0.0))
                        ]
        tram_tfs = [carla.Transform(carla.Location(x=-71.608032, y=26.289885, z=0.5), carla.Rotation(pitch=0, yaw=-90, roll=0.)),
                    carla.Transform(carla.Location(x=226.5, y=97, z=0.5), carla.Rotation(pitch=0, yaw=90, roll=0.)),
                    carla.Transform(carla.Location(x=57, y=210, z=0.5), carla.Rotation(pitch=0, yaw=0, roll=0.)),
                    carla.Transform(carla.Location(x=117, y=12, z=0.5), carla.Rotation(pitch=0, yaw=0, roll=0.))]
    elif "town04" in map_name.lower(): 
        tram_tfs = [carla.Transform(carla.Location(x=201.033508, y=-276.552521, z=0.5), carla.Rotation(pitch=0, yaw=90, roll=0)),
                    carla.Transform(carla.Location(x=332, y=-332, z=0.5), carla.Rotation(pitch=0, yaw=40, roll=0))]
    elif "town05" in map_name.lower():
        train_tfs = [carla.Transform(carla.Location(x=60.044647, y=-231.470352, z=12.096813), carla.Rotation(pitch=0, yaw=-180, roll=0.)),
                    carla.Transform(carla.Location(x=-70.044647, y=-232.470352, z=12.096813), carla.Rotation(pitch=0, yaw=-180, roll=0.)),
                    carla.Transform(carla.Location(x=-140.044647, y=-233.470352, z=12.296813), carla.Rotation(pitch=0, yaw=-180, roll=0.)),
                    carla.Transform(carla.Location(y=0, x=-294.658752, z=12.76813), carla.Rotation(pitch=0, yaw=-90, roll=0.0)),
                    carla.Transform(carla.Location(y=-150, x=-290.658752, z=12.76813), carla.Rotation(pitch=0, yaw=-90, roll=0.0))]
        tram_tfs = [carla.Transform(carla.Location(x=-135, y=61, z=0.5), carla.Rotation(pitch=0, yaw=90, roll=0)),
                    carla.Transform(carla.Location(x=-59.6, y=-42, z=0.5), carla.Rotation(pitch=0, yaw=90, roll=0))]
    elif "town06" in map_name.lower():
        tram_tfs = [carla.Transform(carla.Location(x=20, y=16, z=0.), carla.Rotation(0, 95, 0)),
                    carla.Transform(carla.Location(x=-18, y=194, z=0.), carla.Rotation(0, -90, 0))]
        train_tfs = [carla.Transform(carla.Location(700, 125, .2), carla.Rotation(0, 95, 0)),
                     carla.Transform(carla.Location(370, 129.5, .2), carla.Rotation(0, -180, 0)),
                     carla.Transform(carla.Location(201, 129, .2), carla.Rotation(0, -180, 0)),
                     carla.Transform(carla.Location(433, 61, .2), carla.Rotation(0, -180, 0))]
    elif "town10" in map_name.lower():
        tram_tfs = [carla.Transform(carla.Location(x=34.540833, y=9.891460, z=0.5), carla.Rotation(pitch=0., yaw=180., roll=0.))]
    static_vehicles = []
    if tram_tfs:
        tram_bp = blueprint_library.find(trains[1])
        for t in tram_tfs:
            t_actor = spawn(world,tram_bp,t)
            # if "06" not in map_name.lower():
            #     t_actor.set_simulate_physics(True)
            static_vehicles.append(t_actor)
    
    if train_tfs:
        train_bp = blueprint_library.find(trains[0])
        for t in train_tfs:
            static_vehicles.append(spawn(world,train_bp,t))

    return static_vehicles


def spawn(world, bp, tr):
    tmp = world.get_blueprint_library().filter('walker')[6]
    tmp = world.try_spawn_actor(tmp, carla.Transform(tr.location, carla.Rotation(0,0,0)))
    time.sleep(1)
    if tmp is not None:
        tr.location = tmp.get_transform().location
        tr.location.z -= tmp.bounding_box.extent.z
        tmp.destroy()
    tags_dict = custom_tag(bp)
    v = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,-30), carla.Rotation(0,0,0)))
    v.set_simulate_physics(False)
    v.set_transform(tr)
    if tags_dict is not None:
        v.update_semantic_tags(tags_dict)
    return v


def get_bbox_volume(bbox):
    e = bbox.extent
    v = e.x * e.y * e.z
    return v
 
    
def get_bp_list_from_bbox_volume(bbox_vol):
    if bbox_vol < .5:
        return two_wheel
    if bbox_vol < 3.1: # only broken truck is wolkswagen t2 (bbox ~= 2.5m3) 
        return cars
    return trucks


def get_corners_from_bb(bbox, tr=None):
    x,  y,  z = bbox.location.x, bbox.location.y, bbox.location.z
    ex, ey, ez = bbox.extent.x, bbox.extent.y, bbox.extent.z
    rp, ry, rr = bbox.rotation.pitch, bbox.rotation.yaw, bbox.rotation.roll

    if tr is None:
        rot = R.from_euler('yzx', [rp,ry,rr], degrees=True).as_matrix()
    else:
        rot = R.from_euler('yzx', [rp+tr.rotation.pitch,ry+tr.rotation.yaw,rr+tr.rotation.roll], degrees=True).as_matrix()

    if tr is None:
        center = np.array([x,y,z])
    else:
        center = np.array([x+tr.location.x,y+tr.location.y,z+tr.location.z])

    corners = np.array([
        [ ex,  ey,  ez],
        [ ex,  ey, -ez],
        [ ex, -ey,  ez],
        [ ex, -ey, -ez],
        [-ex,  ey,  ez],
        [-ex,  ey, -ez],
        [-ex, -ey,  ez],
        [-ex, -ey, -ez]
    ])

    return (center + (corners @ rot.T)).tolist()


"""
Classes:

    None         =   0u,
    Buildings    =   1u,
    Fences       =   2u,
    Other        =   3u,
    Pedestrians  =   4u,
    Poles        =   5u,
    RoadLines    =   6u,
    Roads        =   7u,
    Sidewalks    =   8u,
    Vegetation   =   9u,
    Vehicles     =  10u,
    Walls        =  11u,
    TrafficSigns =  12u,
    Sky          =  13u,
    Ground       =  14u,
    Bridge       =  15u,
    RailTrack    =  16u,
    GuardRail    =  17u,
    TrafficLight =  18u,
    Static       =  19u,
    Dynamic      =  20u,
    Water        =  21u,
    Terrain      =  22u,
    Persons      =  40u,
    Riders       =  41u,
    Cars         =  100u,
    Trucks       =  101u,
    Busses       =  102u,
    Trains       =  103u,
    Motorcycles  =  104u,
    Bycicles     =  105u,
    Any          =  0xFF

"""
