# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import isaacgym
import numpy as np
import torch

from oscar import ASSETS_ROOT
from oscar.utils.object_utils import create_cylinder, create_box
from oscar.tasks.agent_task import AgentTask
from isaacgym import gymtorch
from isaacgym import gymapi
from oscar.utils.torch_utils import axisangle2quat, quat2axisangle, rotate_vec_by_quat, to_torch, tensor_clamp, quat_mul


class Pick(AgentTask):
	"""
	Robot Manipulation task that involves pushing a small puck up an inclined table along a specific path
	"""

	def __init__(
			self,
			cfg,
	):
		# Store relevant task information
		task_cfg = cfg["task"]
		self.aggregate_mode = task_cfg["aggregateMode"]

		# reward info
		self.reward_settings = {
			"r_reach_scale": task_cfg["r_reach_scale"],
			"r_contact_scale": task_cfg["r_contact_scale"],
			"r_goal_scale": task_cfg["r_goal_scale"],
			"r_press_scale": task_cfg["r_press_scale"],
		}

		# reset info
		self.reset_settings = {
			"height_threshold": None,           # Height below which reset will occur
		}

		# Other info from config we need
		# Temporary directory for storing generated cup files
		self._tmp_asset_path = "/tmp"
		self.steps_per_vis_update = 1
		self._platform_size = [0.7,0.7,0.05]

		# Placeholders that will be filled in later
		self.cameras = []
		if task_cfg['rgb_render']:
			self.num_cameras = min(task_cfg['num_cameras'], task_cfg['numEnvs'])
		else:	
			self.num_cameras = 0

		# Private placeholders
		# (x,y,z) position of start platform state
		self._platform_surface_pos = None
		# (x,y,z) position of goal platform state
		self._goal_bin_state = None             # (x,y,z) position of goal bin
		self._puck_state = None              # Root body state
		# Density of each puck in each env (scaled)
		self._puck_density = None
		# Friction of each puck in each env (scaled)
		self._puck_friction = None
		# Friction of path in each env (scaled)
		self._puck_size = None              # Size of each puck in each env
		self._init_puck_state = None           # Initial state of puck for the current env
		# Actor ID corresponding to agent for a given env
		self._agent_id = None
		self._puck_id = None                 # Actor ID corresponding to puck for a given env
		self._color_frac = None             # Color fraction values for each env
		self._default_puck_rb_props = None  # Default properties for puck rigid body

		# Run super init
		super().__init__(cfg=cfg)

	def _create_envs(self):
		# Always run super method for create_envs first
		super()._create_envs()

		# Define bounds for env spacing
		lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
		upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

		# Create platform asset
		platform_opts = gymapi.AssetOptions()
		platform_opts.fix_base_link = True
		platform_asset = self.gym.create_box(
			self.sim, *self._platform_size, platform_opts
		)

		# Create table stand asset
		table_stand_height = 0.1
		table_stand_offset = -0.1
		table_stand_pos = [-0.60, 0.0, 1.0 + self._platform_size[2] /
											 2 - table_stand_height / 2 + table_stand_offset]
		table_stand_opts = gymapi.AssetOptions()
		table_stand_opts.fix_base_link = True
		table_stand_asset = self.gym.create_box(
			self.sim, *[0.2, 0.2, table_stand_height], platform_opts)
		self.reset_settings["height_threshold"] = table_stand_pos[2] - 0.1

		# Create puck asset(s)
		self._puck_density = torch.ones(self.n_envs, 1, device=self.device)
		self._puck_friction = torch.ones(self.n_envs, 1, device=self.device)
		self._puck_size = torch.ones(self.n_envs, 1, device=self.device)
		puck_asset = None

		# Define randomized values
		puck_sizes = np.linspace(
			self.task_cfg["puck_size"][0], self.task_cfg["puck_size"][1], self.n_envs)

		# Shuffle ordering for each
		np.random.shuffle(puck_sizes)

		# Iterate over all values to fill in per-env physical parameters
		puck_assets = []
		for i, puck_size in enumerate(puck_sizes):
			# Cube asset
			self._puck_size[i] = puck_size
			puck_opts = gymapi.AssetOptions()
			puck_opts.disable_gravity = False
			puck_opts.collapse_fixed_joints = True
			# Dummy value, this will be immediately overridden
			puck_opts.density = 1.0
			# asset = self.gym.create_box(self.sim, *[puck_size, puck_size, 0.3 * puck_size], puck_opts)
			puck_asset_fpath = create_cylinder(
				name=f"puck",
				size=[0.5 * puck_size, 0.4 * puck_size],
				mass=np.pi * ((0.5 * puck_size) ** 2) * 0.4 * puck_size,
				generate_urdf=True,
				unique_urdf_name=False,
				visual_top_site=False,
				from_mesh=False,
				hollow=False,
				asset_root_path=self._tmp_asset_path,
			)
			asset = self.gym.load_asset(
				self.sim, self._tmp_asset_path, puck_asset_fpath, puck_opts)
			puck_assets.append(asset)

		puck_color = gymapi.Vec3(0.0, 0.7, 0.9)

		# create goal assets
		goal_opts = gymapi.AssetOptions()
		goal_opts.density = 0
		goal_opts.disable_gravity = True
		goal_opts.fix_base_link = True
		# TODO not hard code
		goal_asset = self.gym.load_asset(
			self.sim, '/home/pcy/rl/oscar/assets/urdf/', 'sphere.urdf', goal_opts)

		# Define start pose for agent
		agent_start_pose = gymapi.Transform()
		agent_start_pose.p = gymapi.Vec3(-0.55, 0.0,
																		 1.0 + self._platform_size[2] / 2 + table_stand_offset)
		agent_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

		# Define start pose for start platform
		platform_pos = np.array([0.0, 0.0, 1.0])
		platform_start_pose = gymapi.Transform()
		platform_start_pose.p = gymapi.Vec3(*platform_pos)
		platform_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
		self._platform_surface_pos = np.array(
			platform_pos) + np.array([0, 0, self._platform_size[2] / 2])

		# Define start pose for table stand
		table_stand_start_pose = gymapi.Transform()
		table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
		table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

		# Define dummy start pose (for actors whose positions get overridden during reset() anyways)
		dummy_start_pose = gymapi.Transform()
		dummy_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
		dummy_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

		# compute aggregate size
		n_agent_bodies = self.gym.get_asset_rigid_body_count(self.agent_asset)
		n_agent_shapes = self.gym.get_asset_rigid_shape_count(self.agent_asset)
		n_puck_bodies = self.gym.get_asset_rigid_body_count(puck_assets[0])
		n_puck_shapes = self.gym.get_asset_rigid_shape_count(puck_assets[0])
		n_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
		n_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
		# 1 for table, start / goal platform, path segments
		max_agg_bodies = n_agent_bodies + n_puck_bodies + n_goal_bodies + 3
		# 1 for table, start / goal platform, path segments
		max_agg_shapes = n_agent_shapes + n_puck_shapes + n_goal_shapes + 3

		# Determine number of envs to create
		n_per_row = int(np.sqrt(self.n_envs))

		# Create environments
		self._color_frac = np.zeros(self.n_envs)
		self._default_puck_rb_props = {
			"mass": [],
			"invMass": [],
			"inertia": [],
			"invInertia": [],
		}
		for i in range(self.n_envs):
			# create env instance
			env_ptr = self.gym.create_env(self.sim, lower, upper, n_per_row)

			# Create actors and define aggregate group appropriately depending on setting
			# NOTE: Agent should ALWAYS be loaded first in sim!
			if self.aggregate_mode >= 3:
				self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

			# Create agent
			# Potentially randomize start pose
			if self.agent_pos_noise > 0:
				rand_xy = self.agent_pos_noise * (-1. + np.random.rand(2) * 2.0)
				agent_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
																				 1.0 + self._platform_size[2] / 2 + table_stand_height)
			if self.agent_rot_noise > 0:
				rand_rot = torch.zeros(1, 3)
				rand_rot[:, -1] = self.agent_rot_noise * (-1. + np.random.rand() * 2.0)
				new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
				agent_start_pose.r = gymapi.Quat(*new_quat)
			agent_actor = self.gym.create_actor(
				env_ptr, self.agent_asset, agent_start_pose, self.agent.name, i, 0, 0)
			self.gym.set_actor_dof_properties(
				env_ptr, agent_actor, self.agent_dof_props)

			# Record agent ID if we haven't done so already
			if self._agent_id is None:
				self._agent_id = agent_actor

			if self.aggregate_mode == 2:
				self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

			# Create agent stand and platforms
			platform_actor = self.gym.create_actor(
				env_ptr, platform_asset, platform_start_pose, "platform", i, 1, 0)
			table_stand_actor = self.gym.create_actor(
				env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

			if self.aggregate_mode == 1:
				self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

			# Create pucks
			self._puck_id = self.gym.create_actor(
				env_ptr, puck_assets[i], dummy_start_pose, "puck", i, 2, 0)
			# Store default rigid body props
			puck_rb_props = self.gym.get_actor_rigid_body_properties(
				env_ptr, self._puck_id)[0]
			self._default_puck_rb_props["mass"].append(puck_rb_props.mass)
			self._default_puck_rb_props["invMass"].append(puck_rb_props.invMass)
			self._default_puck_rb_props["inertia"].append(puck_rb_props.inertia)
			self._default_puck_rb_props["invInertia"].append(
				puck_rb_props.invInertia)

			# create goal
			self._goal_id = self.gym.create_actor(
				env_ptr, goal_asset, dummy_start_pose, "goal", i+self.n_envs, 0, 0)

			if self.aggregate_mode > 0:
				self.gym.end_aggregate(env_ptr)

			# Store the created env pointers
			self.envs.append(env_ptr)

		for j in range(self.num_cameras):
			# create camera
			camera_properties = gymapi.CameraProperties()
			camera_properties.width = 320
			camera_properties.height = 200
			h1 = self.gym.create_camera_sensor(self.envs[j], camera_properties)
			camera_position = gymapi.Vec3(1, -1, 1)
			camera_target = gymapi.Vec3(0, 0, 0)
			self.gym.set_camera_location(
				h1, self.envs[j], camera_position, camera_target)
			self.cameras.append(h1)

		# Setup init state buffer
		self._init_puck_state = torch.zeros(self.n_envs, 13, device=self.device)

	def setup_references(self):
		# Always run super method first
		super().setup_references()

		# Fill in relevant handles
		task_handles = {
			"puck": self.gym.find_actor_rigid_body_handle(self.envs[0], self._puck_id, "puck"),
			"goal": self.gym.find_actor_rigid_body_handle(self.envs[0], self._goal_id, "goal"),
		}
		self.handles.update(task_handles)

		# Store tensors to hold states
		self._puck_state = self.sim_states.actor_root_states[:, self._puck_id, :]
		self._goal_state = self.sim_states.actor_root_states[:, self._goal_id, :]
		self.goal = self._goal_state[:, :3]

		# Store other necessary tensors
		self._current_path_segment = torch.zeros(
			self.n_envs, device=self.device, dtype=torch.long)
		self._last_path_segment = torch.zeros_like(self._current_path_segment)

		# Randomize properties
		self._reset_properties()

		# Store references to the contacts
		self.contact_forces.update({
			"puck": self.sim_states.contact_forces[:, self.handles["puck"], :],
		})

	def _compute_rewards(self, actions):
		# Compose dict of contacts
		contacts = {
			"arm": torch.sum(torch.norm(self.contact_forces["arm"], dim=-1), dim=-1),
			"eef": torch.norm(self.contact_forces["eef"], dim=-1),
			"puck": self.contact_forces["puck"],
		}

		# Compute reward (use jit function for speed)
		return _compute_task_rewards(
			actions=actions,
			contacts=contacts,
			states=self.states,
			reward_settings=self.reward_settings,
		)

	def _compute_resets(self):
		# Compute resets (use jit function for speed)
		return _compute_task_resets(
			reset_buf=self.reset_buf,
			progress_buf=self.progress_buf,
			states=self.states,
			max_episode_length=self.max_episode_length,
			reset_settings=self.reset_settings,
		)

	def _update_states(self, dt=None):
		"""
		Updates the internal states for this task (should update self.states dict)

		NOTE: Assumes simulation has already refreshed states!!

		Args:
				dt (None or float): Amount of sim time (in seconds) that has passed since last update. If None, will assume
						that this is simply a refresh (no sim time has passed) and will not update any values dependent on dt
		"""
		# Always run super method first
		super()._update_states(dt=dt)
		env_ids = torch.arange(start=0, end=self.n_envs,
													 device=self.device, dtype=torch.long)

		# Update internal states
		z_vec = torch.zeros_like(self._puck_state[:, :3])
		z_vec[:, 2] = 1.0
		z_rot = rotate_vec_by_quat(vec=z_vec, quat=self._puck_state[:, 3:7])

		self.states.update({
			"puck_quat": self._puck_state[:, 3:7].clone(),
			"puck_pos": self._puck_state[:, :3].clone(),
			"target_pos": self.goal.clone(),
			"puck_tilt": torch.abs(torch.nn.CosineSimilarity(dim=-1)(z_rot, z_vec).unsqueeze(dim=-1)),
			"puck_pos_relative": self._puck_state[:, :3] - self.agent.states["eef_pos"],
		})

	def reset(self, env_ids=None):
		# If env_ids is None, we reset all the envs
		if env_ids is None:
			env_ids = torch.arange(start=0, end=self.n_envs,
														 device=self.device, dtype=torch.long)

		# Reset properties
		self._reset_properties(env_ids=env_ids)

		# We must do a reset here to make sure the changes to the rigid body are propagated correctly
		self.sim_states.set_actor_root_states_indexed(env_ids=env_ids)
		self.sim_states.set_dof_states_indexed(env_ids=env_ids)
		self.sim_states.clear_contact_forces_indexed(env_ids=env_ids)
		self.gym.simulate(self.sim)
		self.sim_states.refresh(contact_forces=False)

		# Reset puck (must occur AFTER resetting properties because something weird happens with states being overridden otherwise)
		self._reset_puck_state(env_ids=env_ids)
		self._reset_goal_state(env_ids=env_ids)

		# Always run super reset at the end
		super().reset(env_ids=env_ids)

	def _reset_puck_state(self, env_ids=None):
		"""
		Simple method to sample @puck's position based on self.startPositionNoise and self.startRotationNoise, and
		automatically reset the pose internally. Populates self._init_puck_state and automatically writes to the
		corresponding self._puck_state

		Args:
				env_ids (tensor or None): Specific environments to reset puck for
		"""
		# If env_ids is None, we reset all the envs
		if env_ids is None:
			env_ids = torch.arange(start=0, end=self.n_envs,
														 device=self.device, dtype=torch.long)

		# Initialize buffer to hold sampled values
		n_resets = len(env_ids)
		sampled_puck_state = torch.zeros(n_resets, 13, device=self.device)

		# Set z value, which is fixed height
		sampled_puck_state[:, 2] = self._platform_surface_pos[2] + \
			self._puck_size[env_ids].squeeze(-1) * 0.4 * 0.5

		# Initialize rotation, which is no rotation (quat w = 1)
		sampled_puck_state[:, 6] = 1.0

		# Sample x y values
		sampled_puck_state[:, :2] = 0.5*(torch.rand(n_resets, 2, dtype=torch.float, device=self.device) - 0.5) * torch.tensor(self._platform_size[:2], device=self.device)

		# Sample rotation value
		if self.start_rotation_noise > 0:
			aa_rot = torch.zeros(n_resets, 3, device=self.device)
			aa_rot[:, 2] = 2.0 * self.start_rotation_noise * \
				(torch.rand(n_resets, device=self.device) - 0.5)
			sampled_puck_state[:, 3:7] = quat_mul(
				axisangle2quat(aa_rot), sampled_puck_state[:, 3:7])

		# Lastly, set these sampled values as the new init state
		self._init_puck_state[env_ids, :] = sampled_puck_state

		# Write to the sim states
		self._puck_state[env_ids] = self._init_puck_state[env_ids]

	def _reset_goal_state(self, env_ids=None):
		if env_ids is None:
			env_ids = torch.arange(start=0, end=self.n_envs,
														 device=self.device, dtype=torch.long)

		# Initialize buffer to hold sampled values
		n_resets = len(env_ids)
		sampled_goal_pos = torch.zeros(n_resets, 3, device=self.device)

		# Set z value, which is fixed height
		sampled_goal_pos[:, 2] = self._platform_surface_pos[2] + \
			self._puck_size[env_ids].squeeze(-1) * 0.4 * 0.5

		# Sample x y values
		sampled_goal_pos[:, :2] = 0.5*(torch.rand(n_resets, 2, dtype=torch.float, device=self.device) - 0.5) * torch.tensor(self._platform_size[:2], device=self.device)

		# Write to the sim states
		self.goal[env_ids] = sampled_goal_pos

	def _reset_properties(self, env_ids=None):
		"""
		Method to reset properties in specific environments specified by @env_ids.

		Args:
				env_ids (tensor or None): Specific environments to reset env properties for
		"""
		# If env_ids is None, we reset all the envs
		if env_ids is None:
			env_ids = torch.arange(start=0, end=self.n_envs,
														 device=self.device, dtype=torch.long)

		# Initialize buffer to hold sampled values
		n_resets = len(env_ids)

		# Reset the puck friction and densities
		f_puck_log_min, f_puck_log_max = np.log10(
			self.task_cfg["puck_friction"][0]), np.log10(self.task_cfg["puck_friction"][1])
		self._puck_friction[env_ids, :] = to_torch(np.power(10., np.random.uniform(
			low=f_puck_log_min,
			high=f_puck_log_max,
			size=(n_resets, 1)
		)), dtype=torch.float, device=self.device)
		d_puck_log_min, d_puck_log_max = np.log10(
			self.task_cfg["puck_density"][0]), np.log10(self.task_cfg["puck_density"][1])
		self._puck_density[env_ids, :] = to_torch(np.power(10., np.random.uniform(
			low=d_puck_log_min,
			high=d_puck_log_max,
			size=(n_resets, 1)
		)), dtype=torch.float, device=self.device)

		# Update in sim
		for env_id in env_ids:
			env = self.envs[env_id]

			# Set color values
			puck_r_frac = (np.log10(self._puck_friction[env_id].item(
			)) - f_puck_log_min) / (f_puck_log_max - f_puck_log_min)
			puck_g_frac = (np.log10(self._puck_density[env_id].item(
			)) - d_puck_log_min) / (d_puck_log_max - d_puck_log_min)
			puck_color = [puck_r_frac, puck_g_frac, 0.0] if (d_puck_log_max != d_puck_log_min) and (
				f_puck_log_max != f_puck_log_min) else [0.7, 0.7, 0.0]

			# Set puck values
			puck_rs_props = self.gym.get_actor_rigid_shape_properties(
				env, self._puck_id)
			puck_rb_props = self.gym.get_actor_rigid_body_properties(
				env, self._puck_id)
			puck_rs_props[0].friction = self._puck_friction[env_id]
			puck_rb_props[0].mass = self._default_puck_rb_props["mass"][env_id] * \
				self._puck_density[env_id].item()
			self.gym.set_actor_rigid_shape_properties(
				env, self._puck_id, puck_rs_props)
			self.gym.set_actor_rigid_body_properties(
				env, self._puck_id, puck_rb_props, recomputeInertia=True)
			self.gym.set_rigid_body_color(
				env, self._puck_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*puck_color))
			self.gym.set_rigid_body_color(
				env, self._goal_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*puck_color))

		# Update extrinsic states
		self.states.update({
			"puck_density": torch.log(self._puck_density),
			"puck_friction": torch.log(self._puck_friction),
			"puck_size": self._puck_size.clone(),
		})

	@property
	def force_sim_step_during_reset(self):
		return False


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def _compute_task_rewards(
		actions,
		contacts,
		states,
		reward_settings,
):
	# type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]) -> Tensor

	# Compute distance from hand to puck
	d_hand = torch.norm(states["puck_pos_relative"], dim=-1)
	reach_reward = (0.5 * (1 - torch.tanh(10.0 * d_hand)) + 0.5 *
									(1 - torch.tanh(10.0 * torch.abs(states["puck_pos_relative"][:, 2]))))

	# Reward for making contact with EEF and any object, with no contact of arm
	# As sanity check, also check for EEF being in relative range of puck
	bad_contact = contacts["arm"] > 0
	good_contact = contacts["eef"] > 0.25
	in_range = d_hand < states["puck_size"].squeeze(-1) * 5.0
	contact_reward = good_contact & ~bad_contact & in_range

	# Penalty for pressing too hard
	puck_contact = contacts["puck"]
	press_penalty = torch.abs(puck_contact[:, 2]) > 10.0

	# Compute distance from puck to target position
	d_target = torch.norm(states["target_pos"] - states["puck_pos"], dim=-1)
	target_reward = 1 - torch.tanh(10.0 * d_target)

	# We provide the path success reward + maximum between (target reward + reach reward + contact reward, goal reward)
	rewards = \
		reward_settings["r_press_scale"] * press_penalty + \
		torch.max(
			0.5 * target_reward * reward_settings["r_goal_scale"] +
			reward_settings["r_reach_scale"] * reach_reward +
			contact_reward * reward_settings["r_contact_scale"],
		)

	return rewards


@torch.jit.script
def _compute_task_resets(
		reset_buf,
		progress_buf,
		states,
		max_episode_length,
		reset_settings,
):
	# type: (Tensor, Tensor, Dict[str, Tensor], int, Dict[str, float]) -> Tensor

	reset_buf = torch.where(
		(progress_buf >= max_episode_length - 1),
		torch.ones_like(reset_buf),
		reset_buf
	)

	return reset_buf


if __name__ == '__main__':
	from rl_games.common import vecenv, env_configurations
	from oscar.utils.config import set_seed, get_args, parse_sim_params, load_cfg
	from oscar.utils.parse_task import parse_task
	from attrdict import AttrDict
	args = AttrDict({
		'cfg_env': '/home/pcy/rl/oscar/oscar/cfg/train/base.yaml',
		'cfg_env_add': [
			'/home/pcy/rl/oscar/oscar/cfg/train/sim/physx.yaml',
			'/home/pcy/rl/oscar/oscar/cfg/train/agent/franka.yaml',
			'/home/pcy/rl/oscar/oscar/cfg/train/task/pick.yaml',
			'/home/pcy/rl/oscar/oscar/cfg/train/controller/oscar.yaml',
			'/home/pcy/rl/oscar/oscar/cfg/train/common/eval.yaml'
		],
		'device': 'GPU',
		'num_envs': 1,
		'test': True,
		'train': False,
		'randomize': False,
		'deterministic': True,
		'save_video': False,
		'no_force_sim_gpu': False,
		'max_iterations': 0,
		'seed': 0,
		'ppo_device': 0,
		'resume': False,
		'checkpoint': "Base",
		'logdir': '/home/pcy/rl/oscar/log/',
		'num_test_episodes': 10,
		'episode_length': 0,
		'headless': False,
		'pretrained_delan': '/home/pcy/rl/oscar/trained_models/train/Push/Push_oscar__seed_1.pth',
		'experiment_name': 'debug',
		'num_threads': 0,
		'slices': 0,
		'subscenes': 0,
		'physics_engine': gymapi.SIM_PHYSX,
		'use_gpu': True,
	})
	cfg, logdir = load_cfg(args, use_rlg_config=True)
	sim_params = parse_sim_params(args, cfg)
	task, env = parse_task(args, cfg["env"], sim_params)
	for _ in range(1000):
		env.step(torch.tensor(env.act_space.sample(), device='cuda:0'))
		print(env.render(mode='rgb_array'))