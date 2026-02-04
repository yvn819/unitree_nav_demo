ENABLE_PATH_VIZ = True

# CSV path relative to this file's directory.
PATH_VIZ_CSV = "path/wall_waypoints.csv"

# If True, shift CSV waypoints so the first point starts at the robot's
# current base position. If False, use CSV coordinates as-is.
PATH_VIZ_RELATIVE_TO_START = True

# Optional offset added after any relative shift.
PATH_VIZ_OFFSET = [0.0, 0.0, 0.0]

# Visualization parameters
PATH_VIZ_COLOR = [1.0, 0.1, 0.1, 1.0]  # RGBA
PATH_VIZ_WIDTH = 0.01
PATH_VIZ_Z_OFFSET = 0.02
