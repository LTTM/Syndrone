import carla
import time
import sys
from tqdm import trange
from pathlib import Path
from modules import utils, carla_utils


if __name__ == "__main__":
    print("\n=== LOG SPECTATOR SCRIPT ===\n")

    # Connect to CARLA
    client = carla_utils.CarlaClient()
    try:
        client.connect()
    except RuntimeError as e:
        print("Could not connect to CARLA server:\n\t", e)
        sys.exit(1)

    # Ask the user which town to load
    accepted_val = ["01", "02", "03", "04", "05", "06", "07", "10"]
    opt_accepted_val = ["1", "2", "3", "4", "5", "6", "7"]
    print("Which town you wont to load?")
    print("Available towns:")
    print("\t- Town01;")
    print("\t- Town02;")
    print("\t- Town03;")
    print("\t- Town04;")
    print("\t- Town05;")
    print("\t- Town06;")
    print("\t- Town07;")
    print("\t- Town10;")
    while(True):
        choice = input("\nInsert just the number:  ")
        if choice in accepted_val or choice in opt_accepted_val:
            break
        else:
            print("You have inserted a wrong option, retry.")

    # Ask the user the length of the acquisition
    while(True):
        slen = input("\nInsert the length of the acquisition in seconds:  ")
        try:
            slen = int(slen)
            if slen > 0:
                break
            else:
                print("You have inserted a wrong option, retry.")
        except:
            print("You have inserted a wrong option, retry.")

    # Defien the world to use
    world_number = "0" + choice if choice in opt_accepted_val else choice
    world_number = choice + "HD" if choice == "10" else world_number
    world_name = "Town" + world_number + "_Opt"

    # Define the out file path
    file_name = f"keyframes_{world_name}_{slen}.csv"
    folder_name = Path("data/trajectories")
    folder_name.mkdir(parents=True, exist_ok=True)  # Create the data folder if not already present
    file_path = Path(folder_name / file_name)
    if file_path.exists():
        print("\nThe selected output file already exists")
        print("Do you want to overwrite it? [y/n]")
        while True:
            choice = input()
            if choice == "y":
                break
            elif choice == "n":
                sys.exit(1)
            else:
                print("You have inserted a wrong option, retry.")

    # Load the correct world
    print()
    utils.Loader = utils.Loader("Loading the selected Town...", "Done!", 0.05).start()
    world = client.load_world(name=world_name, weather=carla.WeatherParameters.ClearNoon)
    utils.Loader.stop()

    # Choose the starting point
    input("\nChoose your starting position and press enter\n")
    for i in range(3, 0, -1):
        print("The acquisition wil start in %d" %i, end="\r")
        time.sleep(1)
    print("", end="\r")

    # Load the spectator
    spectator = world.get_spectator()

    # Defien the acquisition parameters
    freq = 10  # fps

    # initialize z0
    z0 = 0
    # Perform the acquisition
    with open(file_path, "w") as f:
        pbar = trange(slen*freq, desc="Aquiring position's keyframes")
        for i in pbar:
            tr = spectator.get_transform()
            gp = world.ground_projection(tr.location)
            if gp.label in [carla.CityObjectLabel.Roads, carla.CityObjectLabel.Ground]: # maybe leave only road -- so we are sure:
                z0 = gp.location.z
            x, y, z = tr.location.x, tr.location.y, tr.location.z
            pitch, yaw, roll = tr.rotation.pitch, tr.rotation.yaw, tr.rotation.roll
            sout = "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"%(x,y,z,pitch,yaw,roll,z0)
            pbar.set_description(f"Aquiring position's keyframes (x: {round(x, 2)}; y: {round(y, 2)}, z0: {round(z0, 2)})")
            f.write(sout + '\n')
            time.sleep(1/freq)

    print("\nacquisition completed!")
