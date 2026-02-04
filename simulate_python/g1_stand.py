import time
from typing import List, Tuple

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

import config


# G1 29-DOF joint order (from unitree_robots/g1/g1_joint_index_dds.md)
class G1JointIndex:
    # legs
    L_HIP_PITCH = 0
    L_HIP_ROLL = 1
    L_HIP_YAW = 2
    L_KNEE = 3
    L_ANKLE_PITCH = 4
    L_ANKLE_ROLL = 5
    R_HIP_PITCH = 6
    R_HIP_ROLL = 7
    R_HIP_YAW = 8
    R_KNEE = 9
    R_ANKLE_PITCH = 10
    R_ANKLE_ROLL = 11
    # waist
    WAIST_YAW = 12
    WAIST_ROLL = 13
    WAIST_PITCH = 14
    # left arm
    L_SHOULDER_PITCH = 15
    L_SHOULDER_ROLL = 16
    L_SHOULDER_YAW = 17
    L_ELBOW = 18
    L_WRIST_ROLL = 19
    L_WRIST_PITCH = 20
    L_WRIST_YAW = 21
    # right arm
    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_SHOULDER_YAW = 24
    R_ELBOW = 25
    R_WRIST_ROLL = 26
    R_WRIST_PITCH = 27
    R_WRIST_YAW = 28


def build_stand_pose() -> List[float]:
    # Basic, conservative stand pose (radians). Tweak as needed.
    pose = [0.0] * 29

    # Legs: deeper, more stable squat
    # pose[G1JointIndex.L_HIP_PITCH] = -0.25
    # pose[G1JointIndex.L_HIP_ROLL] = 0.00
    # pose[G1JointIndex.L_HIP_YAW] = 0.00
    # pose[G1JointIndex.L_KNEE] = 0.5
    # pose[G1JointIndex.L_ANKLE_PITCH] = -0.5
    # pose[G1JointIndex.L_ANKLE_ROLL] = -0.00

    # pose[G1JointIndex.R_HIP_PITCH] = -0.25
    # pose[G1JointIndex.R_HIP_ROLL] = 0.00
    # pose[G1JointIndex.R_HIP_YAW] = 0.00
    # pose[G1JointIndex.R_KNEE] = 0.5
    # pose[G1JointIndex.R_ANKLE_PITCH] = -0.5
    # pose[G1JointIndex.R_ANKLE_ROLL] = 0.00

    # pose[G1JointIndex.WAIST_YAW] = 0.0
    # pose[G1JointIndex.WAIST_ROLL] = 0.0
    # pose[G1JointIndex.WAIST_PITCH] = 0.2

    # pose[G1JointIndex.L_SHOULDER_PITCH] = 0.5
    # pose[G1JointIndex.L_SHOULDER_ROLL] = 0.1
    # pose[G1JointIndex.L_SHOULDER_YAW] = 0.0
    # pose[G1JointIndex.L_ELBOW] = 0.6
    # pose[G1JointIndex.L_WRIST_ROLL] = 0.0
    # pose[G1JointIndex.L_WRIST_PITCH] = 0.0
    # pose[G1JointIndex.L_WRIST_YAW] = 0.0

    # pose[G1JointIndex.R_SHOULDER_PITCH] = 0.5
    # pose[G1JointIndex.R_SHOULDER_ROLL] = -0.1
    # pose[G1JointIndex.R_SHOULDER_YAW] = 0.0
    # pose[G1JointIndex.R_ELBOW] = 0.6
    # pose[G1JointIndex.R_WRIST_ROLL] = 0.0
    # pose[G1JointIndex.R_WRIST_PITCH] = 0.0
    # pose[G1JointIndex.R_WRIST_YAW] = 0.0

    return pose


def build_kp_kd() -> Tuple[List[float], List[float]]:
    # Higher gains for legs, lower for arms/waist.
    kp = [0.0] * 29
    kd = [0.0] * 29

    leg_kp = 120.0
    leg_kd = 3.0
    arm_kp = 30.0
    arm_kd = 1.0
    waist_kp = 50.0
    waist_kd = 2.0

    for i in range(12):
        kp[i] = leg_kp
        kd[i] = leg_kd

    kp[G1JointIndex.WAIST_YAW] = waist_kp
    kp[G1JointIndex.WAIST_ROLL] = waist_kp
    kp[G1JointIndex.WAIST_PITCH] = waist_kp
    kd[G1JointIndex.WAIST_YAW] = waist_kd
    kd[G1JointIndex.WAIST_ROLL] = waist_kd
    kd[G1JointIndex.WAIST_PITCH] = waist_kd

    for i in range(G1JointIndex.L_SHOULDER_PITCH, G1JointIndex.R_WRIST_YAW + 1):
        kp[i] = arm_kp
        kd[i] = arm_kd

    return kp, kd


def main() -> None:
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()

    cmd = unitree_hg_msg_dds__LowCmd_()
    num_motor_cmd = len(cmd.motor_cmd)

    stand_pose = build_stand_pose()
    kp, kd = build_kp_kd()

    dt = 0.002
    ramp_time = 2.0
    start_time = time.monotonic()

    while True:
        t = time.monotonic() - start_time
        ratio = np.clip(t / ramp_time, 0.0, 1.0)

        cmd.mode_pr = 0
        cmd.mode_machine = 0

        for i in range(num_motor_cmd):
            if i < len(stand_pose):
                cmd.motor_cmd[i].mode = 1
                cmd.motor_cmd[i].q = ratio * stand_pose[i]
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kp = kp[i]
                cmd.motor_cmd[i].kd = kd[i]
                cmd.motor_cmd[i].tau = 0.0
            else:
                cmd.motor_cmd[i].mode = 0
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(dt)


if __name__ == "__main__":
    main()
