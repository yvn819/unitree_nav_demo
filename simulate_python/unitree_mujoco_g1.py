import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config
try:
    import path_viz_config as path_viz
except Exception:
    path_viz = None


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

def _init_base_path(current_base_pos: np.ndarray):
    csv_rel = getattr(config, "BASE_WAYPOINTS_CSV", "")
    waypoints = []

    if csv_rel:
        base_dir = Path(getattr(config, "__file__", Path.cwd())).resolve().parent
        csv_path = base_dir / csv_rel
        if csv_path.exists():
            data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim == 1 and data.size >= 2:
                data = data.reshape(1, -1)
            if data.size > 0:
                for row in data:
                    x = float(row[0])
                    y = float(row[1])
                    yaw = float(row[2]) if row.size > 2 else 0.0
                    waypoints.append(np.array([x, y, yaw], dtype=float))

    if len(waypoints) == 0:
        waypoints = [np.array(p, dtype=float) for p in config.BASE_WAYPOINTS]

    if len(waypoints) == 0:
        waypoints = [np.array([0.0, 0.0, 0.0], dtype=float)]

    stride = int(getattr(config, "BASE_WAYPOINT_STRIDE", 1))
    if stride > 1 and len(waypoints) > 1:
        waypoints = waypoints[::stride]

    # Optionally shift waypoints so the first point starts at current base pos.
    if getattr(config, "BASE_WAYPOINTS_RELATIVE_TO_START", False):
        if len(waypoints) > 0:
            first = waypoints[0].copy()
            for i in range(len(waypoints)):
                waypoints[i][0:2] = waypoints[i][0:2] - first[0:2] + current_base_pos[0:2]

    offset = np.array(getattr(config, "BASE_WAYPOINTS_OFFSET", [0.0, 0.0, 0.0]), dtype=float)
    if np.linalg.norm(offset) > 0.0:
        for i in range(len(waypoints)):
            waypoints[i][0:2] = waypoints[i][0:2] + offset[0:2]

    return waypoints


def _get_base_adrs():
    if not hasattr(_get_base_adrs, "base_x_adr"):
        def _qpos_adr(name: str) -> int:
            adr = mj_model.joint(name).qposadr
            # qposadr can be array-like in some versions; normalize to int
            return int(np.array(adr).ravel()[0])

        _get_base_adrs.base_x_adr = _qpos_adr("base_x")
        _get_base_adrs.base_y_adr = _qpos_adr("base_y")
        _get_base_adrs.base_yaw_adr = _qpos_adr("base_yaw")
    return _get_base_adrs.base_x_adr, _get_base_adrs.base_y_adr, _get_base_adrs.base_yaw_adr


def _get_base_dof_adrs():
    if not hasattr(_get_base_dof_adrs, "base_x_adr"):
        def _qvel_adr(name: str) -> int:
            adr = mj_model.joint(name).dofadr
            # dofadr can be array-like in some versions; normalize to int
            return int(np.array(adr).ravel()[0])

        _get_base_dof_adrs.base_x_adr = _qvel_adr("base_x")
        _get_base_dof_adrs.base_y_adr = _qvel_adr("base_y")
        _get_base_dof_adrs.base_yaw_adr = _qvel_adr("base_yaw")
    return _get_base_dof_adrs.base_x_adr, _get_base_dof_adrs.base_y_adr, _get_base_dof_adrs.base_yaw_adr


def _load_viz_waypoints(current_base_pos: np.ndarray):
    if path_viz is None:
        return []

    csv_rel = getattr(path_viz, "PATH_VIZ_CSV", "")
    waypoints = []

    if csv_rel:
        base_dir = Path(getattr(path_viz, "__file__", Path.cwd())).resolve().parent
        csv_path = base_dir / csv_rel
        if csv_path.exists():
            data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim == 1 and data.size >= 2:
                data = data.reshape(1, -1)
            if data.size > 0:
                for row in data:
                    x = float(row[0])
                    y = float(row[1])
                    yaw = float(row[2]) if row.size > 2 else 0.0
                    waypoints.append(np.array([x, y, yaw], dtype=float))

    if len(waypoints) == 0:
        return []

    stride = int(getattr(path_viz, "PATH_VIZ_STRIDE", getattr(config, "BASE_WAYPOINT_STRIDE", 1)))
    if stride > 1 and len(waypoints) > 1:
        waypoints = waypoints[::stride]

    if getattr(path_viz, "PATH_VIZ_RELATIVE_TO_START", False):
        first = waypoints[0].copy()
        for i in range(len(waypoints)):
            waypoints[i][0:2] = waypoints[i][0:2] - first[0:2] + current_base_pos[0:2]

    offset = np.array(getattr(path_viz, "PATH_VIZ_OFFSET", [0.0, 0.0, 0.0]), dtype=float)
    if np.linalg.norm(offset) > 0.0:
        for i in range(len(waypoints)):
            waypoints[i][0:2] = waypoints[i][0:2] + offset[0:2]

    return waypoints


def _update_path_viz():
    if path_viz is None or not getattr(path_viz, "ENABLE_PATH_VIZ", False):
        return

    if not hasattr(_update_path_viz, "waypoints"):
        base_x_adr, base_y_adr, _ = _get_base_adrs()
        current = np.array(
            [
                mj_data.qpos[base_x_adr],
                mj_data.qpos[base_y_adr],
                0.0,
            ],
            dtype=float,
        )
        _update_path_viz.waypoints = _load_viz_waypoints(current)

    waypoints = _update_path_viz.waypoints
    if len(waypoints) < 2:
        return

    scn = viewer.user_scn
    scn.ngeom = 0
    maxgeom = scn.maxgeom
    max_segments = maxgeom
    stride = max(1, int(np.ceil((len(waypoints) - 1) / max_segments)))
    color = np.array(getattr(path_viz, "PATH_VIZ_COLOR", [1.0, 0.1, 0.1, 1.0]), dtype=float)
    width = float(getattr(path_viz, "PATH_VIZ_WIDTH", 0.01))
    zoff = float(getattr(path_viz, "PATH_VIZ_Z_OFFSET", 0.0))

    for i in range(0, len(waypoints) - 1, stride):
        if scn.ngeom >= maxgeom:
            break
        p0 = waypoints[i].copy()
        p1 = waypoints[min(i + stride, len(waypoints) - 1)].copy()
        p0[2] = zoff
        p1[2] = zoff

        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            color,
        )
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            p0[0], p0[1], p0[2],
            p1[0], p1[1], p1[2],
        )
        scn.ngeom += 1


def _advance_base_along_path(dt: float):
    # Move planar base joints (base_x, base_y, base_yaw) along waypoints using velocity control.
    if not hasattr(_advance_base_along_path, "waypoints"):
        base_x_adr, base_y_adr, base_yaw_adr = _get_base_adrs()
        _advance_base_along_path.base_x_adr = base_x_adr
        _advance_base_along_path.base_y_adr = base_y_adr
        _advance_base_along_path.base_yaw_adr = base_yaw_adr
        vel_x_adr, vel_y_adr, vel_yaw_adr = _get_base_dof_adrs()
        _advance_base_along_path.vel_x_adr = vel_x_adr
        _advance_base_along_path.vel_y_adr = vel_y_adr
        _advance_base_along_path.vel_yaw_adr = vel_yaw_adr
        _advance_base_along_path.pos = np.array(
            [
                mj_data.qpos[base_x_adr],
                mj_data.qpos[base_y_adr],
                0.0,
            ],
            dtype=float,
        )
        _advance_base_along_path.waypoints = _init_base_path(
            _advance_base_along_path.pos
        )
        _advance_base_along_path.index = 0
        _advance_base_along_path.target = _advance_base_along_path.waypoints[0]

    pos = _advance_base_along_path.pos
    target = _advance_base_along_path.target
    delta = target[0:2] - pos[0:2]
    dist = float(np.linalg.norm(delta))
    if dist < 5e-2:
        _advance_base_along_path.index = (
            _advance_base_along_path.index + 1
        ) % len(_advance_base_along_path.waypoints)
        _advance_base_along_path.target = _advance_base_along_path.waypoints[
            _advance_base_along_path.index
        ]
        if getattr(config, "PRINT_GOAL_UPDATES", True):
            reached = target
            next_target = _advance_base_along_path.target
            print(
                "goal reached: x={:.3f} y={:.3f} yaw={:.3f} -> next: x={:.3f} y={:.3f} yaw={:.3f}".format(
                    float(reached[0]),
                    float(reached[1]),
                    float(reached[2]) if reached.size >= 3 else 0.0,
                    float(next_target[0]),
                    float(next_target[1]),
                    float(next_target[2]) if next_target.size >= 3 else 0.0,
                )
            )
        target = _advance_base_along_path.target
        delta = target[0:2] - pos[0:2]
        dist = float(np.linalg.norm(delta))

    # Command planar velocities toward target.
    speed = float(getattr(config, "BASE_SPEED", 3.0))
    dir_xy = delta / max(dist, 1e-6)
    vx = dir_xy[0] * speed
    vy = dir_xy[1] * speed

    # Yaw control: use waypoint yaw if provided, otherwise optional auto-follow.
    yaw_now = float(mj_data.qpos[_advance_base_along_path.base_yaw_adr])
    if target.size >= 3:
        yaw_target = float(target[2])
    elif getattr(config, "BASE_YAW_FOLLOW", False):
        yaw_target = float(np.arctan2(delta[1], delta[0]))
    else:
        yaw_target = yaw_now

    def _wrap_pi(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    yaw_err = _wrap_pi(yaw_target - yaw_now)
    yaw_rate_max = float(getattr(config, "BASE_YAW_RATE", 2.0))
    yaw_rate = np.clip(yaw_err * float(getattr(config, "BASE_YAW_KP", 2.0)), -yaw_rate_max, yaw_rate_max)

    mj_data.qvel[_advance_base_along_path.vel_x_adr] = vx
    mj_data.qvel[_advance_base_along_path.vel_y_adr] = vy
    mj_data.qvel[_advance_base_along_path.vel_yaw_adr] = yaw_rate

    # Update cached position estimate (Mujoco will integrate, but we keep a local copy for distance check).
    pos[0:2] = mj_data.qpos[_advance_base_along_path.base_x_adr:_advance_base_along_path.base_x_adr + 2]

def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

        _advance_base_along_path(mj_model.opt.timestep)
        mujoco.mj_step(mj_model, mj_data)

        do_print = False
        if getattr(config, "PRINT_BASE_STATE", False):
            now = time.perf_counter()
            if not hasattr(SimulationThread, "last_print"):
                SimulationThread.last_print = 0.0
            if now - SimulationThread.last_print >= float(
                getattr(config, "PRINT_BASE_STATE_PERIOD", 0.1)
            ):
                base_x_adr, base_y_adr, base_yaw_adr = _get_base_adrs()
                vel_x_adr, vel_y_adr, vel_yaw_adr = _get_base_dof_adrs()
                base_state = (
                    float(mj_data.qpos[base_x_adr]),
                    float(mj_data.qpos[base_y_adr]),
                    float(mj_data.qpos[base_yaw_adr]),
                    float(mj_data.qvel[vel_x_adr]),
                    float(mj_data.qvel[vel_y_adr]),
                    float(mj_data.qvel[vel_yaw_adr]),
                )
                SimulationThread.last_print = now
                do_print = True

        locker.release()

        if do_print:
            x, y, yaw, vx, vy, yaw_rate = base_state
            print(
                "base: x={:.3f} y={:.3f} yaw={:.3f} | v: vx={:.3f} vy={:.3f} yaw_rate={:.3f}".format(
                    x, y, yaw, vx, vy, yaw_rate
                )
            )

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        _update_path_viz()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
