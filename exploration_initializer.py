import sys
import json
import os
import yaml
import subprocess

high_fov = 60
low_fov = 5

high_range = 30
low_range = 10

num_robots = 0
robots = {}

interpreter_path =  "E:/Summer Research 2023/DME-DRL Daniel/DME_DRL_CO/venv/Scripts/python.exe"
working_directory = "E:/Summer Research 2023/MADDPG_New/MADDPG/src/"
def split_range(start, end, segment_size):
    segments = []
    current = start

    while current < end:
        segments.append(current)
        current += segment_size

    return segments


if len(sys.argv) < 5:
    print("Usage: python exploration_initializer.py <json_file_path> <base_config_file_path> <run_config_file_path> <run_exploration_folder_path>")
    sys.exit(1)

json_file_path = os.path.abspath(sys.argv[1])
base_config_yaml_file_path = os.path.abspath(sys.argv[2])
run_config_yaml_file_path = os.path.abspath(sys.argv[3])
run_config_file_yaml_name = os.path.basename(run_config_yaml_file_path)

print(run_config_yaml_file_path)

try:
    with open(json_file_path, 'r') as json_file:
        best_crew = json.load(json_file)
except FileNotFoundError:
    print(f"File not found: {json_file_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Invalid JSON file: {json_file_path}")
    sys.exit(1)

with open(base_config_yaml_file_path, "r") as yaml_file:
    try:
        # Parse the YAML content
        base_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # Now 'data' contains the parsed YAML content as a Python dictionary
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
robotRadius = base_config["default_robot"]["robotRadius"]
commRange = base_config["default_robot"]["commRange"]
syncRange = base_config["default_robot"]["syncRange"]
resetRandomPose = base_config["default_robot"]["resetRandomPose"]
startPose_x = base_config["default_robot"]["startPose"]["x"]
startPose_y = base_config["default_robot"]["startPose"]["y"]
resolution = base_config["default_robot"]["resolution"]
w1 = base_config["default_robot"]["w1"]
w2 = base_config["default_robot"]["w2"]
w3 = base_config["default_robot"]["w3"]

for i in range(0, len(best_crew)):
    num_robots += best_crew[i]
color_list = split_range(10, 255, (255 - 10) / num_robots)
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
    num_robots  = best_crew[i]
    for j in range(0,int(num_robots)):
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
    result = subprocess.run([interpreter_path, f'{working_directory}/main.py', run_config_file_yaml_name], check=True, cwd=working_directory, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    result = result.stdout.splitlines()[-1] # The standard output of the subprocess
    print(result)
    # Print or use the captured information as needed



except subprocess.CalledProcessError as e:
    print(f"Error running exploration script_path: {e}")
# base_config

