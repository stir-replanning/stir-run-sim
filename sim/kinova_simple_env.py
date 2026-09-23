#!/usr/bin/env python3
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from scipy.spatial.transform import Rotation as R
import robosuite
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from robosuite.models.objects import BoxObject

from kinova_circle_env import KinovaTwoObjCircle
from kinova_grid_env import make_object, _CUBE_RGBA, CUBE_HALF


class KinovaTwoObjSimple(KinovaTwoObjCircle):

    def __init__(self, guideline_xy=(0.0, -0.10), guideline_size=0.10,
                 cube_half=None, success_speed=0.04, success_rest_z=0.05, **kwargs):
        self._guideline_xy = (float(guideline_xy[0]), float(guideline_xy[1]))
        self._guideline_size = float(guideline_size)
        self._cube_half = float(cube_half) if cube_half is not None else CUBE_HALF
        self._success_rest_z = float(success_rest_z)
        kwargs.setdefault("basket_xy", self._guideline_xy)
        super().__init__(success_speed=success_speed, **kwargs)

    def _make_cube(self, name, inst_name):
        if name in _CUBE_RGBA:
            return BoxObject(name=inst_name, size=[self._cube_half] * 3,
                             rgba=_CUBE_RGBA[name], density=500, friction=[1.0, 0.05, 0.001])
        return make_object(name, inst_name)

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_guideline(mujoco_arena)

        self._objs = [
            self._make_cube(self._obj_names[0], "obj0"),
            self._make_cube(self._obj_names[1], "obj1"),
        ]
        self._basket = None
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self._objs,
        )

    def _add_guideline(self, arena):
        gx, gy = self._guideline_xy
        gz = float(self.table_offset[2]) + 0.001
        half = self._guideline_size / 2.0
        bar = 0.004
        green = [0.10, 0.63, 0.47, 1.0]
        fill = [0.10, 0.63, 0.47, 0.18]

        def add(name, pos, size, rgba):
            g = ET.Element("geom")
            g.set("name", name); g.set("type", "box")
            g.set("pos", array_to_string(pos)); g.set("size", array_to_string(size))
            g.set("rgba", array_to_string(rgba))
            g.set("contype", "0"); g.set("conaffinity", "0"); g.set("group", "1")
            arena.worldbody.append(g)

        add("guideline_fill", [gx, gy, gz], [half, half, 0.0005], fill)
        add("guideline_n", [gx, gy + half, gz], [half, bar, 0.0008], green)
        add("guideline_s", [gx, gy - half, gz], [half, bar, 0.0008], green)
        add("guideline_e", [gx + half, gy, gz], [bar, half, 0.0008], green)
        add("guideline_w", [gx - half, gy, gz], [bar, half, 0.0008], green)

    def _setup_references(self):
        super(Lift, self)._setup_references()
        if self._init_qpos is not None:
            self.robots[0].init_qpos = np.array(self._init_qpos)
        self.obj_body_ids = [self.sim.model.body_name2id(o.root_body) for o in self._objs]
        self.basket_body_id = None

    def _reset_internal(self):
        super(Lift, self)._reset_internal()
        self._home_eef_quat_wxyz = self._site_quat_wxyz(self.robots[0].eef_site_id)
        li = int(self._spec.get("obj_left_idx", 0))
        pos = self.object_positions(self._spec)
        self._place_free(self._objs[li], self.obj_body_ids[li], pos["A"])
        self._place_free(self._objs[1 - li], self.obj_body_ids[1 - li], pos["B"])
        self.sim.forward()
        for _ in range(self._settle_steps):
            self.sim.step()

    def _site_quat_wxyz(self, site_id):
        m = self.sim.model._model
        d = self.sim.data._data
        mujoco.mj_kinematics(m, d)
        q = R.from_matrix(d.site_xmat[site_id].reshape(3, 3).copy()).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def solve_arm_ik(self, target_pos, target_quat_wxyz=None,
                     iters=400, tol=6e-4, lam=0.15, step=0.6):
        m, d = self.sim.model._model, self.sim.data._data
        sid = self.robots[0].eef_site_id
        qadr = np.asarray(self.robots[0]._ref_joint_pos_indexes, int)
        vadr = np.asarray(self.robots[0]._ref_joint_vel_indexes, int)
        if target_quat_wxyz is None:
            target_quat_wxyz = getattr(self, "_home_eef_quat_wxyz", None)
        tR = R.from_quat([target_quat_wxyz[1], target_quat_wxyz[2],
                          target_quat_wxyz[3], target_quat_wxyz[0]])
        limited = m.jnt_limited[qadr].astype(bool)
        lo, hi = m.jnt_range[qadr, 0], m.jnt_range[qadr, 1]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        perr_norm = 1e9
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            cur_pos = d.site_xpos[sid].copy()
            cur_R = R.from_matrix(d.site_xmat[sid].reshape(3, 3).copy())
            perr = target_pos - cur_pos
            rerr = (tR * cur_R.inv()).as_rotvec()
            perr_norm = float(np.linalg.norm(perr))
            if perr_norm < tol and np.linalg.norm(rerr) < 2e-2:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, vadr], jacr[:, vadr]])
            e = np.concatenate([perr, rerr])
            dq = step * (J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e))
            q = d.qpos[qadr] + dq
            d.qpos[qadr] = np.where(limited, np.clip(q, lo, hi), q)
        return d.qpos[qadr].copy(), perr_norm

    def set_arm_to_grasp(self, obj_xy, grasp_z, quat_wxyz=None):
        qsol, perr = self.solve_arm_ik(np.array([obj_xy[0], obj_xy[1], grasp_z]), quat_wxyz)
        d = self.sim.data._data
        d.qpos[np.asarray(self.robots[0]._ref_joint_pos_indexes, int)] = qsol
        d.qvel[np.asarray(self.robots[0]._ref_joint_vel_indexes, int)] = 0.0
        self.sim.forward()
        return perr

    def _collision_geom_sets(self):
        if getattr(self, "_cube_geom_ids", None) is None:
            m = self.sim.model
            self._cube_geom_ids = set(g for g in range(m.ngeom)
                                      if m.geom_bodyid[g] in set(self.obj_body_ids))
            self._robot_geom_ids = set(g for g in range(m.ngeom)
                                       if (m.geom_id2name(g) or "").startswith(("robot0", "gripper0")))
        return self._cube_geom_ids, self._robot_geom_ids

    def gripper_hits_block(self, depth=0.001):
        m, d = self.sim.model._model, self.sim.data._data
        mujoco.mj_forward(m, d)
        cubes, robot = self._collision_geom_sets()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            hit = (g1 in cubes and g2 in robot) or (g2 in cubes and g1 in robot)
            if hit and c.dist < depth:
                return True
        return False

    @property
    def guideline_xy(self):
        return self._guideline_xy

    @property
    def guideline_size(self):
        return self._guideline_size

    def guideline_pos(self):
        return np.array([self._guideline_xy[0], self._guideline_xy[1],
                         float(self.table_offset[2]) + 0.001])

    def basket_pos(self):
        return self.guideline_pos()

    def _check_success(self):
        op = self.target_pos()
        gx, gy = self._guideline_xy
        half = self._guideline_size / 2.0
        inside = (abs(op[0] - gx) <= half) and (abs(op[1] - gy) <= half)
        rest_z = float(self.table_offset[2]) + 0.002 + self._success_rest_z
        released = op[2] < rest_z
        settled = self.target_speed() < self._success_speed
        return bool(inside and released and settled)


def make_simple_env(view_size, obj_names=("red_cube", "yellow_cube"),
                    obj_center=(0.0, 0.10), obj_sep=0.05, obj_axis="x",
                    guideline_xy=(0.0, -0.10), guideline_size=0.10,
                    cube_half=0.02, extra_cams=()):
    controller = load_controller_config(default_controller="OSC_POSE")
    cams = ["agentview", "robot0_eye_in_hand"] + list(extra_cams)
    env = robosuite.make(
        env_name="KinovaTwoObjSimple",
        robots="Kinova3",
        gripper_types="MenagerieRobotiq2f85",
        controller_configs=controller,
        obj_names=obj_names,
        obj_center=obj_center,
        obj_sep=obj_sep,
        obj_axis=obj_axis,
        guideline_xy=guideline_xy,
        guideline_size=guideline_size,
        cube_half=cube_half,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cams,
        camera_heights=view_size,
        camera_widths=view_size,
        control_freq=20,
        ignore_done=True,
        horizon=100000,
        reward_shaping=False,
    )
    return env


if __name__ == "__main__":
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from scipy.spatial.transform import Rotation as R
    env = make_simple_env(256)
    for li, tc in [(0, 0), (1, 0)]:
        env.set_episode(li, tc)
        env.reset()
        print(f"obj_left={li} target_cell={tc} target={env.target_name} "
              f"@ {np.round(env.target_pos(),3)}  cellA={np.round(env.cellA_xy,3)} "
              f"cellB={np.round(env.cellB_xy,3)}  guideline={np.round(env.guideline_pos(),3)} "
              f"size={env.guideline_size}")
    print("success (start):", env._check_success())
    gp = env.guideline_pos()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    env.sim.data.set_joint_qpos(env._objs[env._spec["target_idx"]].joints[0],
                                np.concatenate([[gp[0], gp[1], gp[2] + 0.026], q]))
    env.sim.forward()
    for _ in range(80):
        env.sim.step()
    print("target after place:", np.round(env.target_pos(), 3))
    print("success (placed in guideline):", env._check_success())
    os._exit(0)
