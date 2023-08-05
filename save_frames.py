import sys
from modules import utils, carla_utils, helpers


if __name__ == "__main__":
    print("\n=== SAVE FRAME SCRIPT ===\n")

    # Load the args
    args = utils.ArgParser(script="save_frames").parse()
    town_name = args.town
    pov = args.pov
    weather = args.weather
    slen = args.slen
    fps = args.fps
    use_lidar = args.lidar
    dry = args.dry

    # Print the selected options
    print("--------------------------------------------------------------------")
    print(f"The following options have been selected:")
    print(f"The rendering will be performed on the world: {town_name}")
    print(f"The following POVs will be used: {pov}")
    print("--------------------------------------------------------------------\n")

    # Define the CARLA client
    client = carla_utils.CarlaClient()
    try:
        client.connect()
    except RuntimeError as e:
        print("Could not connect to CARLA server:\n\t", e)
        sys.exit(1)

    # Define and generate the output folders
    try:
        folders = helpers.out_folders_gen(
            town_name=town_name,
            weather=weather,
            slen=slen,
            pov=pov,
            use_lidar=use_lidar,
            verbose=True,
        )
    except ValueError as e:
        print(e.args[0])

    # Perform the rendering
    client.render_scene(
        folders=folders,
        town_name=town_name,
        pov=pov,
        weather=weather,
        slen=slen,
        fps=fps,
        use_lidar=use_lidar,
        dry=dry,
        verbose="full",
    )
