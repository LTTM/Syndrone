import random
import carla
from .custom_bp_tags import custom_tag
from tqdm import trange
import time

def spawn_walkers(world, N=200, verbose=True):
    #loader = Loader("Spawning...", "Done!", 0.05).start()
    pedestrians = []
    controllers = []
    bp_lib = world.get_blueprint_library()
    wai_bp = bp_lib.find('controller.ai.walker')
    if verbose:
        pbar = trange(N, desc="Trying to spawn... Progress")
    else:
        pbar = range(N)
    for _ in pbar:
        spawn = carla.Transform()
        spawn.location = world.get_random_location_from_navigation()
        if spawn.location is not None:
            spawn.location.z += 1
            bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
            actor = world.try_spawn_actor(bp, spawn)
            if actor is not None:
                tags_dict = custom_tag(bp)
                if tags_dict is not None:
                    actor.update_semantic_tags(tags_dict)
                con = world.try_spawn_actor(wai_bp, spawn, actor)
                con.go_to_location(world.get_random_location_from_navigation())
                con.set_max_speed(1 + random.random())
                time.sleep(.2)
                con.start()
                pedestrians.append(actor)
                controllers.append(con)
    #loader.stop()
    print("Spawned %d Walkers"%len(pedestrians))
    return pedestrians, controllers


def spawn_vehicles(world, manager, N=300, verbose=True):
    #loader = Loader("Spawning...", "Done!", 0.05).start()
    vehicles = []
    ways = world.get_map().generate_waypoints(4.0)
    if verbose:
        pbar = trange(N, desc="Trying to spawn... Progress")
    else:
        pbar = range(N)
    for _ in pbar:
        bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
        # don't spawn any trains 
        # only 1/6 of the trucks
        # only 1/2 of the bikes
        while ('train' in bp.id) or \
                    ('truck' in bp.id and random.random() < .85) or \
                        ('bicycle' in bp.id and random.random() < .5):
            bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
        spawn = random.choice(ways).transform
        if spawn.location is not None:
            actor = world.try_spawn_actor(bp, spawn)
            if actor is not None:
                tags_dict = custom_tag(bp)
                if tags_dict is not None:
                    actor.update_semantic_tags(tags_dict)
                actor.set_autopilot(True, manager.get_port())
                vehicles.append(actor)
    #loader.stop()
    print("Spawned %d Vehicles"%len(vehicles))
    return vehicles
