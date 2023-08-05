import carla
import sys
from tqdm import tqdm
from modules import utils, carla_utils, helpers


# Define the possible values for the WEATHER, DAYTIME and TOWN
WEATHERS = [
            "Clear",
            #"Cloudy",
            #"HardRain",
            #"MidFoggy"
           ]
DAYTIMES = [
            "Noon",
            #"Sunset",
            #"Night"
           ]
TOWNS = [
         #"Town01_Opt",
         #"Town02_Opt",
         #"Town03_Opt",
         #"Town04_Opt",
         #"Town05_Opt",
         "Town06_Opt",
         #"Town07_Opt",
         #"Town10HD_Opt"
        ]


if __name__ == "__main__":
    # Defien all the possible permutations of the WEATHER and DAYTIME
    scenarios = []
    for weather in WEATHERS:
        for daytime in DAYTIMES:
            scenarios.append(
                (weather + daytime, carla.WeatherParameters.__dict__[weather + daytime])
            )

    # Load the args
    args = utils.ArgParser(script="run_simulation").parse()
    pov = args.pov
    slen = args.slen
    fps = args.fps
    use_lidar = args.lidar
    dry = args.dry

    # Run the simulation for each scenario
    pbar = tqdm(TOWNS, desc="Running the simulations")
    for town in TOWNS:
        pbar.set_description(
            f"Running the simulations for {town}"
        )  # Update the progress bar description
        inner_pbar = tqdm(scenarios, desc="Rendering")
        for scenario in inner_pbar:
            # Define the CARLA client
            client = carla_utils.CarlaClient(timeout=3000)
            try:
                client.connect()
            except RuntimeError as e:
                print("Could not connect to CARLA server:\n\t", e)
                sys.exit(1)

            # Define and generate the output folders
            try:
                folders = helpers.out_folders_gen(
                    town_name=town,
                    weather=scenario,
                    slen=slen,
                    pov=pov,
                    use_lidar=use_lidar,
                    verbose=False,
                )
            except ValueError as e:
                print(e.args[0])

            inner_pbar.set_description(
                f"Rendering the {scenario[0]} scenario"
            )  # Update the progress bar description

            # Perform the rendering
            client.render_scene(
                folders=folders,
                town_name=town,
                pov=pov,
                weather=scenario,
                slen=slen,
                fps=fps,
                use_lidar=use_lidar,
                dry=dry,
                master_pbar=inner_pbar,
                verbose="all" #"minimal"
            )
