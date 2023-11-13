import sys
import json
import os
import yaml
import subprocess




# working_directory = "D:/2023 Summer/MADDPG/src"
multi_agent_working_directory = "E:/Summer Research 2023/MADDPG_New/MADDPG/src/"
single_agent_working_directory = "E:/Summer Research 2023/MADDPG_single_agent/MADDPG/src/"
# interpreter_path = "C:/Users/Daniel Yin/AppData/Local/Programs/Python/Python39/python.exe"
interpreter_path = "E:/Summer Research 2023/DME-DRL Daniel/DME_DRL_CO/venv/Scripts/python.exe"



def split_range(start, end, segment_size):
    segments = []
    current = start

    while current < end:
        segments.append(current)
        current += segment_size

    return segments

def main():
    high_fov = 60
    low_fov = 5

    high_range = 30
    low_range = 10

    num_robots = 0
    robots = {}
    if len(sys.argv) < 4:
        print("Usage: python exploration_initializer.py <json_file_path> <base_config_file_path> <run_config_file_path> <run_exploration_folder_path>")
        print(len(sys.argv))

        sys.exit(1)

    json_file_path = os.path.abspath(sys.argv[1])
    base_config_yaml_file_path = os.path.abspath(sys.argv[2])
    run_config_yaml_folder_path = os.path.abspath(sys.argv[3])

    try:
        with open(json_file_path, 'r') as json_file:
            best_crew = json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {json_file_path}")
        sys.exit(1)

    # Assuming best_crew is a list/tuple with at least 4 elements
    run_config_file_yaml_name = "config_{}.yaml".format(''.join(map(str, best_crew[:4])))

    # Construct the full path and then get the basename
    run_config_yaml_file_path = os.path.join(run_config_yaml_folder_path, run_config_file_yaml_name)


    with open(base_config_yaml_file_path, "r") as yaml_file:
        try:
            # Parse the YAML content
            base_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
            # Now 'data' contains the parsed YAML content as a Python dictionary
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")

    cache_file_path = base_config["cache_file_path"]



    directory = os.path.dirname(cache_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(cache_file_path):
        with open(cache_file_path, 'w') as file:
            json.dump([], file)


    with open(cache_file_path, 'r') as file:
        data = json.load(file)

    # Search for the target list and retrieve its value
    for entry in data:
        if entry['list'] == best_crew:
            print(entry['value'])
            return
    else:
        print(f"The list {best_crew} was not found in the JSON data.")


    robotRadius = base_config["default_robot"]["robotRadius"]
    commRange = base_config["default_robot"]["commRange"]
    syncRange = base_config["default_robot"]["syncRange"]
    resetRandomPose = base_config["default_robot"]["resetRandomPose"]
    resolution = base_config["default_robot"]["resolution"]
    w1 = base_config["default_robot"]["w1"]
    w2 = base_config["default_robot"]["w2"]
    w3 = base_config["default_robot"]["w3"]

    for i in range(0, len(best_crew)):
        num_robots += best_crew[i]
    if(num_robots>1):
        color_list = split_range(10, 255, (255 - 10) / num_robots)
    else:
        color_list = [100]
    print(color_list)
    robot_index = 0
    for i in range(0, len(best_crew)):
        fov = 0
        laser_range = 0
        if(i == 0):
            fov = high_fov
            laser_range = high_range
        elif (i == 1):
            fov = high_fov
            laser_range = low_range
        elif (i == 2):
            fov = low_fov
            laser_range = high_range
        elif (i == 3):
            fov = low_fov
            laser_range = low_range


        num_robots  = best_crew[i]
        if(num_robots!=0):
            for j in range(0,int(num_robots)):
                startPose_x = base_config["default_robot"]["startPose"]["robot" + str((i*3)+j+1)]["x"]
                startPose_y = base_config["default_robot"]["startPose"]["robot" + str((i*3)+j+1)]["y"]
                robot = {
                    "color": color_list[robot_index],
                    "robotRadius": robotRadius,
                    "resetRandomPose": resetRandomPose,
                    "startPose": {
                        "x": startPose_x,
                        "y": startPose_y
                    },
                    "laser": {
                        "range": laser_range,
                        "fov": fov,
                        "resolution": resolution
                    }
                }
                robots[f'robot{robot_index+1}'] = robot
                robot_index += 1


    robots["commRange"] = commRange
    robots["syncRange"] = syncRange
    robots["number"] = robot_index
    robots["w1"] = w1
    robots["w2"] = w2
    robots["w3"] = w3

    base_config["robots"] = robots
    print(base_config)
    with open(run_config_yaml_file_path, 'w') as yaml_file:
        yaml.dump(base_config, yaml_file)

    try:
        if(robot_index == 1):
            result = subprocess.run(
                [interpreter_path, f'{single_agent_working_directory}/main.py', './BO_TO_MADDPG/'+run_config_file_yaml_name], check=True,
                cwd=single_agent_working_directory, stdout=subprocess.PIPE, text=True, encoding='utf-8')
            result = result.stdout.splitlines()[-1]  # The standard output of the subprocess
            print(result)
        else:
            result = subprocess.run([interpreter_path, f'{multi_agent_working_directory}/main.py', './BO_TO_MADDPG/'+run_config_file_yaml_name], check=True, cwd=multi_agent_working_directory, stdout=subprocess.PIPE, text=True, encoding='utf-8')
            result = result.stdout.splitlines()[-1]  # The standard output of the subprocess
            print(result)
        # Print or use the captured information as needed

    # Create the file (or overwrite if it already exists)


    except subprocess.CalledProcessError as e:
        print(f"Error running exploration script_path: {e}")

    new_data = {"list": best_crew, "value": result}

    try:
        with open(cache_file_path, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):  # Ensure data is a list
                data = []
    except (FileNotFoundError, json.JSONDecodeError):  # Handle case where file doesn't exist or is empty/malformed
        data = []



    # Append the new data
    data.append(new_data)

    # Write the updated data back to the JSON file
    with open(cache_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return

if __name__ == "__main__":
    main()

# base_config

