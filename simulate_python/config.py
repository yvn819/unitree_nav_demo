ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene_29dof.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # disable for navigation
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.002  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

ENABLE_ELASTIC_BAND: 1 # Virtual spring band, used for lifting h1

PRINT_BASE_STATE = True
PRINT_BASE_STATE_PERIOD = 0.1  # seconds

BASE_WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]

# Optional CSV path for waypoints (relative to this file's directory).
# CSV format: header "x,y,z" and numeric rows.
BASE_WAYPOINTS_CSV = "path/wall_waypoints.csv"

# If True, shift CSV waypoints so the first point starts at the robot's
# current base position. If False, use CSV coordinates as-is.
BASE_WAYPOINTS_RELATIVE_TO_START = True

# Optional offset added after any relative shift.
BASE_WAYPOINTS_OFFSET = [0.0, 0.0, 0.0]
