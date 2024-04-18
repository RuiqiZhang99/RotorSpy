from tkinter.tix import Tree
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "ax",
    "ay",
    "az",
    "tau1",
    "tau2",
    "tau3",
    "thrust1",
    "thrust2",
    "thrust3",
    "thrust4",
    "debug1",
    "act1",
    "act2",
    "act3",
    "act4",
    "cmdwx",
    "cmdwy",
    "cmdwz",
    "proper_acc_x",
    "proper_acc_y",
    "proper_acc_z",
    "cmd_proper_acc_z",
    "payload_mass_ratio_in_ep",
    "goal_posx",
    "goal_posy",
    "goal_posz",
    "ext_torque_x",
    "ext_torque_y",
    "ext_torque_z",
    "goal_tau1",
    "goal_tau2",
    "goal_tau3",
    "pri_motor_omega1",
    "pri_motor_omega2",
    "pri_motor_omega3",
    "pri_motor_omega4",
    "motor_omega1",
    "motor_omega2",
    "motor_omega3",
    "motor_omega4",
    "priv_act1",
    "priv_act2",
    "priv_act3",
    "priv_act4"
]

def test_policy(env, policy, iterations):
    traj_df = pd.DataFrame(columns=columns)
    max_ep_length = 499
    success = np.zeros(shape=(iterations, 1))
    for itr in range(iterations):
        ep_success = True
        obs = env.reset(random=True)
        episode_id = np.array([itr].reshape((1,1)))
        for step in range(max_ep_length):
            act, _ = policy.predict(obs, deterministic=True)
            act = np.array(act, dtype=np.float64)
            obs, rew, done, info = env.step(act)


            state = env.getQuadState()
            goalstate = env.getQuadGoalState()
            action = env.getQuadAct()
            priv_action = env.getQuadPrivAct()

            # reshape vector
            done = done[:, np.newaxis]
            rew = rew[:, np.newaxis]

            # stack all the data
            data = np.hstack((episode_id, done, rew, state, action, goalstate, priv_action))
            data_frame = pd.DataFrame(data=data, columns=columns)

            # append trajectory
            traj_df = pd.concat([traj_df, data_frame], axis=0, ignore_index=True)

            if done[0]:
                ep_success = False
                print("Episode Failed")
                break
        if ep_success:
            success[itr] = 1
    return traj_df, success


def traj_rollout(env, policy):
    traj_df = pd.DataFrame(columns=columns)
    max_ep_length = 10000
    obs = env.reset(random=False)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    for _ in range(max_ep_length):
        act, _ = policy.predict(obs, deterministic=True)
        act = np.array(act, dtype=np.float64)
        #
        obs, rew, done, info = env.step(act)

        episode_id[done] += 1

        state = env.getQuadState()
        goalstate = env.getQuadGoalState()
        action = env.getQuadAct()
        priv_action = env.getQuadPrivAct()

        # reshape vector
        done = done[:, np.newaxis]
        rew = rew[:, np.newaxis]

        # stack all the data
        data = np.hstack((episode_id, done, rew, state, action, goalstate, priv_action))
        data_frame = pd.DataFrame(data=data, columns=columns)

        # append trajectory
        traj_df = pd.concat([traj_df, data_frame], axis=0, ignore_index=True)
    return traj_df




def plot3d_traj(ax3d, pos, vel):
    sc = ax3d.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=np.linalg.norm(vel, axis=1),
        cmap="jet",
        s=1,
        alpha=0.5,
    )
    ax3d.view_init(elev=40, azim=50)
    #
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    # ax3d.set_zticks([])

    #
    # ax3d.get_proj = lambda: np.dot(
    # Axes3D.get_proj(ax3d), np.diag([1.0, 1.0, 1.0, 1.0]))
    # zmin, zmax = ax3d.get_zlim()
    # xmin, xmax = ax3d.get_xlim()
    # ymin, ymax = ax3d.get_ylim()
    # x_f = 1
    # y_f = (ymax - ymin) / (xmax - xmin)
    # z_f = (zmax - zmin) / (xmax - xmin)
    # ax3d.set_box_aspect((x_f, y_f * 2, z_f * 2))


def test_policy(env, model, render=False):
    max_ep_length = env.max_episode_steps
    num_rollouts = 5
    frame_id = 0
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            # print(obs)
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)

            #
            env.render(ep_len)

            # ======Gray Image=========
            # gray_img = np.reshape(
            #     env.getImage()[0], (env.img_height, env.img_width))
            # cv2.imshow("gray_img", gray_img)
            # cv2.waitKey(100)

            # ======RGB Image=========
            # img =env.getImage(rgb=True)
            # rgb_img = np.reshape(
            #    img[0], (env.img_height, env.img_width, 3))
            # cv2.imshow("rgb_img", rgb_img)
            # os.makedirs("./images", exist_ok=True)
            # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img)
            # cv2.waitKey(100)

            # # # ======Depth Image=========
            # depth_img = np.reshape(env.getDepthImage()[
            #                        0], (env.img_height, env.img_width))
            # os.makedirs("./depth", exist_ok=True)
            # cv2.imwrite("./depth/img_{0:05d}.png".format(frame_id), depth_img.astype(np.uint16))
            # cv2.imshow("depth", depth_img)
            # cv2.waitKey(100)

            #
            ep_len += 1
            frame_id += 1

    #
    if render:
        env.disconnectUnity()
