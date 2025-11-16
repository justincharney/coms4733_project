#!/usr/bin/env python3
"""
Evaluation metrics for robotic grasping model performance.

This module provides metrics to evaluate the performance of
grasping models including task success, collision detection, trajectory
smoothness, and control quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class EpisodeMetrics:
    """
    Tracks and computes metrics for a single episode.

    This class collects data during episode execution and computes
    various performance metrics at the end of the episode.
    """

    def __init__(
        self,
        bin_target_pos: np.ndarray = np.array([0.6, 0.0, 1.15]),
        bin_region_size: np.ndarray = np.array([0.2, 0.3, 0.1]),
        position_tolerance: float = 0.05,
        dt: float = 0.002,
    ):
        """
        Initialize episode metrics tracker.

        Args:
            bin_target_pos: Target position for cube deposition [x, y, z]
            bin_region_size: Size of bin region for success check [x, y, z]
            position_tolerance: Tolerance for settling time calculation (meters)
            dt: Simulation timestep (seconds)
        """
        self.bin_target_pos = np.array(bin_target_pos)
        self.bin_region_size = np.array(bin_region_size)
        self.position_tolerance = position_tolerance
        self.dt = dt

        # Episode data storage
        self.cube_positions: List[np.ndarray] = []
        self.joint_positions: List[np.ndarray] = []
        self.joint_velocities: List[np.ndarray] = []
        self.joint_accelerations: List[np.ndarray] = []
        self.contacts: List[Dict] = []
        self.timestamps: List[float] = []
        self.rewards: List[float] = []

        # Computed metrics (set after episode ends)
        self.task_success: Optional[bool] = None
        self.collision_rate: Optional[float] = None
        self.final_position_error: Optional[float] = None
        self.mean_jerk: Optional[float] = None
        self.rms_joint_acceleration: Optional[float] = None
        self.overshoot: Optional[float] = None
        self.settling_time: Optional[float] = None

    def add_step(
        self,
        cube_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        contacts: Dict,
        reward: float = 0.0,
        timestamp: Optional[float] = None,
    ):
        """
        Add data from a single simulation step.

        Args:
            cube_pos: Current cube position [x, y, z]
            joint_pos: Current joint positions (array)
            joint_vel: Current joint velocities (array)
            contacts: Dictionary with contact information
            reward: Reward received at this step
            timestamp: Optional timestamp (defaults to step * dt)
        """
        self.cube_positions.append(np.array(cube_pos))
        self.joint_positions.append(np.array(joint_pos))
        self.joint_velocities.append(np.array(joint_vel))
        self.contacts.append(contacts)
        self.rewards.append(reward)

        if timestamp is None:
            timestamp = len(self.timestamps) * self.dt
        self.timestamps.append(timestamp)

        # Compute acceleration from velocity differences
        if len(self.joint_velocities) >= 2:
            accel = (self.joint_velocities[-1] - self.joint_velocities[-2]) / self.dt
            self.joint_accelerations.append(accel)
        else:
            self.joint_accelerations.append(np.zeros_like(joint_vel))

    def compute_metrics(self):
        """Compute all metrics from collected episode data."""
        if len(self.cube_positions) == 0:
            raise ValueError("No episode data collected. Cannot compute metrics.")

        # Task success rate
        self.task_success = self._check_task_success()

        # Collision rate
        self.collision_rate = self._compute_collision_rate()

        # Final position error
        self.final_position_error = self._compute_final_position_error()

        # Trajectory smoothness
        self.mean_jerk, self.rms_joint_acceleration = self._compute_smoothness()

        # Overshoot
        self.overshoot = self._compute_overshoot()

        # Settling time
        self.settling_time = self._compute_settling_time()

    def _check_task_success(self) -> bool:
        """
        Check if cube is deposited within bin region at termination.

        Returns:
            True if cube is within bin region, False otherwise
        """
        if len(self.cube_positions) == 0:
            return False

        final_pos = self.cube_positions[-1]
        bin_min = self.bin_target_pos - self.bin_region_size / 2
        bin_max = self.bin_target_pos + self.bin_region_size / 2

        # Check if cube is within bin region
        in_bin = np.all(final_pos >= bin_min) and np.all(final_pos <= bin_max)

        # Check if a successful grasp occurred (reward > 0)
        successful_grasp = any(r > 0 for r in self.rewards)

        return in_bin and successful_grasp

    def _compute_collision_rate(self) -> float:
        """
        Compute collision rate: proportion of steps with collisions.

        Collisions are defined as contacts with table or non-target objects.
        Target objects are the cube being grasped.

        Returns:
            Collision rate (0.0 to 1.0)
        """
        if len(self.contacts) == 0:
            return 0.0

        collision_steps = 0
        for contact_info in self.contacts:
            # Check if there are any unwanted contacts
            if contact_info.get('has_collision', False):
                collision_steps += 1

        return collision_steps / len(self.contacts) if len(self.contacts) > 0 else 0.0

    def _compute_final_position_error(self) -> float:
        """
        Compute Euclidean distance from cube final pose to bin target.

        Returns:
            Final position error in meters
        """
        if len(self.cube_positions) == 0:
            return float('inf')

        final_pos = self.cube_positions[-1]
        error = np.linalg.norm(final_pos - self.bin_target_pos)
        return float(error)

    def _compute_smoothness(self) -> Tuple[float, float]:
        """
        Compute trajectory smoothness metrics.

        Returns:
            Tuple of (mean_jerk, rms_joint_acceleration)
        """
        if len(self.joint_accelerations) < 2:
            return 0.0, 0.0

        # Compute jerk (derivative of acceleration)
        jerks = []
        for i in range(1, len(self.joint_accelerations)):
            jerk = (self.joint_accelerations[i] - self.joint_accelerations[i - 1]) / self.dt
            jerks.append(jerk)

        if len(jerks) == 0:
            return 0.0, 0.0

        # Mean jerk (average magnitude across all joints and time)
        mean_jerk = np.mean([np.linalg.norm(j) for j in jerks])

        # RMS joint acceleration
        all_accelerations = np.array(self.joint_accelerations)
        rms_accel = np.sqrt(np.mean(all_accelerations ** 2))

        return float(mean_jerk), float(rms_accel)

    def _compute_overshoot(self) -> float:
        """
        Compute overshoot: amount trajectory goes past target before correcting.

        Returns:
            Maximum overshoot distance in meters
        """
        if len(self.cube_positions) == 0:
            return 0.0

        # Compute distance to target at each step
        distances_to_target = [
            np.linalg.norm(pos - self.bin_target_pos)
            for pos in self.cube_positions
        ]

        # Find minimum distance (closest approach to target)
        min_distance = min(distances_to_target)
        min_idx = distances_to_target.index(min_distance)

        # Check if we went past the target (overshoot)
        # Overshoot occurs if we get close then move away
        if min_idx < len(distances_to_target) - 1:
            # After closest approach, check if distance increases
            overshoot = 0.0
            for i in range(min_idx + 1, len(distances_to_target)):
                if distances_to_target[i] > min_distance:
                    overshoot = max(overshoot, distances_to_target[i] - min_distance)
            return float(overshoot)

        return 0.0

    def _compute_settling_time(self) -> float:
        """
        Compute settling time: time until position error enters and remains
        within tolerance band.

        Returns:
            Settling time in seconds, or None if never settles
        """
        if len(self.cube_positions) == 0:
            return float('inf')

        # Compute position error at each step
        position_errors = [
            np.linalg.norm(pos - self.bin_target_pos)
            for pos in self.cube_positions
        ]

        # Find first time error enters tolerance and stays within
        settling_idx = None
        consecutive_within_tolerance = 0
        required_consecutive = max(10, len(position_errors) // 10)  # At least 10% of episode

        for i, error in enumerate(position_errors):
            if error <= self.position_tolerance:
                consecutive_within_tolerance += 1
                if consecutive_within_tolerance >= required_consecutive:
                    settling_idx = i - required_consecutive + 1
                    break
            else:
                consecutive_within_tolerance = 0

        if settling_idx is not None:
            return self.timestamps[settling_idx]
        else:
            return float('inf')  # Never settled

    def get_metrics_dict(self) -> Dict:
        """
        Get all computed metrics as a dictionary.

        Returns:
            Dictionary containing all metrics
        """
        return {
            'task_success': self.task_success,
            'collision_rate': self.collision_rate,
            'final_position_error': self.final_position_error,
            'mean_jerk': self.mean_jerk,
            'rms_joint_acceleration': self.rms_joint_acceleration,
            'overshoot': self.overshoot,
            'settling_time': self.settling_time,
        }

    def reset(self):
        """Reset all collected data for a new episode."""
        self.cube_positions.clear()
        self.joint_positions.clear()
        self.joint_velocities.clear()
        self.joint_accelerations.clear()
        self.contacts.clear()
        self.timestamps.clear()
        self.rewards.clear()

        self.task_success = None
        self.collision_rate = None
        self.final_position_error = None
        self.mean_jerk = None
        self.rms_joint_acceleration = None
        self.overshoot = None
        self.settling_time = None


class MetricsEvaluator:
    """
    Evaluates model performance across multiple episodes.

    This class manages multiple EpisodeMetrics instances and computes
    aggregate statistics across episodes.
    """

    def __init__(
        self,
        bin_target_pos: np.ndarray = np.array([0.6, 0.0, 1.15]),
        bin_region_size: np.ndarray = np.array([0.2, 0.3, 0.1]),
        position_tolerance: float = 0.05,
        dt: float = 0.002,
    ):
        """
        Initialize metrics evaluator.

        Args:
            bin_target_pos: Target position for cube deposition
            bin_region_size: Size of bin region for success check
            position_tolerance: Tolerance for settling time
            dt: Simulation timestep
        """
        self.bin_target_pos = bin_target_pos
        self.bin_region_size = bin_region_size
        self.position_tolerance = position_tolerance
        self.dt = dt

        self.episodes: List[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None

    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode = EpisodeMetrics(
            bin_target_pos=self.bin_target_pos,
            bin_region_size=self.bin_region_size,
            position_tolerance=self.position_tolerance,
            dt=self.dt,
        )

    def add_step(
        self,
        cube_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        contacts: Dict,
        reward: float = 0.0,
        timestamp: Optional[float] = None,
    ):
        """
        Add step data to current episode.

        Args:
            cube_pos: Current cube position
            joint_pos: Current joint positions
            joint_vel: Current joint velocities
            contacts: Contact information
            reward: Reward at this step
            timestamp: Optional timestamp
        """
        if self.current_episode is None:
            self.start_episode()

        self.current_episode.add_step(
            cube_pos, joint_pos, joint_vel, contacts, reward, timestamp
        )

    def end_episode(self) -> Dict:
        """
        End current episode and compute metrics.

        Returns:
            Dictionary of metrics for the completed episode
        """
        if self.current_episode is None:
            raise ValueError("No episode in progress. Call start_episode() first.")

        self.current_episode.compute_metrics()
        metrics = self.current_episode.get_metrics_dict()
        self.episodes.append(self.current_episode)
        self.current_episode = None

        return metrics

    def get_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all completed episodes.

        Returns:
            Dictionary containing aggregate statistics
        """
        if len(self.episodes) == 0:
            return {}

        # Task success rate
        success_count = sum(1 for ep in self.episodes if ep.task_success)
        task_success_rate = success_count / len(self.episodes)

        # Collision rate (average across episodes)
        collision_rates = [ep.collision_rate for ep in self.episodes if ep.collision_rate is not None]
        avg_collision_rate = np.mean(collision_rates) if collision_rates else 0.0

        # Final position error (average)
        position_errors = [ep.final_position_error for ep in self.episodes if ep.final_position_error is not None]
        avg_position_error = np.mean(position_errors) if position_errors else float('inf')

        # Trajectory smoothness (average)
        mean_jerks = [ep.mean_jerk for ep in self.episodes if ep.mean_jerk is not None]
        avg_mean_jerk = np.mean(mean_jerks) if mean_jerks else 0.0

        rms_accels = [ep.rms_joint_acceleration for ep in self.episodes if ep.rms_joint_acceleration is not None]
        avg_rms_accel = np.mean(rms_accels) if rms_accels else 0.0

        # Overshoot (average)
        overshoots = [ep.overshoot for ep in self.episodes if ep.overshoot is not None]
        avg_overshoot = np.mean(overshoots) if overshoots else 0.0

        # Settling time (average, excluding infinite values)
        settling_times = [
            ep.settling_time for ep in self.episodes
            if ep.settling_time is not None and ep.settling_time != float('inf')
        ]
        avg_settling_time = np.mean(settling_times) if settling_times else float('inf')

        return {
            'num_episodes': len(self.episodes),
            'task_success_rate': task_success_rate,
            'avg_collision_rate': avg_collision_rate,
            'avg_final_position_error': avg_position_error,
            'avg_mean_jerk': avg_mean_jerk,
            'avg_rms_joint_acceleration': avg_rms_accel,
            'avg_overshoot': avg_overshoot,
            'avg_settling_time': avg_settling_time,
        }

    def print_summary(self):
        """Print a formatted summary of aggregate metrics."""
        metrics = self.get_aggregate_metrics()

        print("\n" + "=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)
        print(f"Number of Episodes: {metrics['num_episodes']}")
        print(f"\nTask Success Rate: {metrics['task_success_rate']:.2%}")
        print(f"Collision Rate (target 0): {metrics['avg_collision_rate']:.4f}")
        print(f"Average Final Position Error: {metrics['avg_final_position_error']:.4f} m")
        print(f"Average Mean Jerk: {metrics['avg_mean_jerk']:.4f}")
        print(f"Average RMS Joint Acceleration: {metrics['avg_rms_joint_acceleration']:.4f}")
        print(f"Average Overshoot: {metrics['avg_overshoot']:.4f} m")
        if metrics['avg_settling_time'] != float('inf'):
            print(f"Average Settling Time: {metrics['avg_settling_time']:.4f} s")
        else:
            print("Average Settling Time: Never settled")
        print("=" * 60 + "\n")


def extract_contact_info(sim) -> Dict:
    """
    Extract contact information from MuJoCo simulation.

    Args:
        sim: MuJoCo simulation object (mjSim)

    Returns:
        Dictionary with contact information
    """
    # Get number of contacts
    ncon = sim.data.ncon

    has_collision = ncon > 0

    # Get contact details (simplified - you may want to filter by body names)
    contact_bodies = []
    if has_collision:
        for i in range(ncon):
            contact = sim.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = sim.model.geom_bodyid[geom1]
            body2 = sim.model.geom_bodyid[geom2]
            body1_name = sim.model.body_id2name(body1)
            body2_name = sim.model.body_id2name(body2)
            contact_bodies.append((body1_name, body2_name))

    return {
        'has_collision': has_collision,
        'n_contacts': ncon,
        'contact_bodies': contact_bodies,
    }


def get_cube_position(sim, cube_body_name: str = "pick_box") -> np.ndarray:
    """
    Get current cube position from simulation.

    Args:
        sim: MuJoCo simulation object
        cube_body_name: Name of the cube body in the XML

    Returns:
        Cube position [x, y, z]
    """
    try:
        body_id = sim.model.body_name2id(cube_body_name)
        pos = sim.data.body_xpos[body_id].copy()
        return pos
    except (ValueError, KeyError):
        # Fallback: try to find any object with "box" in name
        for i in range(sim.model.nbody):
            body_name = sim.model.body_id2name(i)
            if "box" in body_name.lower() and "drop" not in body_name.lower():
                pos = sim.data.body_xpos[i].copy()
                return pos
        # If nothing found, return zeros
        return np.array([0.0, 0.0, 0.0])


def get_joint_states(sim, joint_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get current joint positions and velocities from simulation.

    Args:
        sim: MuJoCo simulation object
        joint_ids: Optional list of joint IDs to extract (None = all actuated joints)

    Returns:
        Tuple of (joint_positions, joint_velocities)
    """
    if joint_ids is None:
        # Get all actuated joints
        joint_pos = sim.data.qpos.copy()
        joint_vel = sim.data.qvel.copy()
    else:
        joint_pos = np.array([sim.data.qpos[i] for i in joint_ids])
        joint_vel = np.array([sim.data.qvel[i] for i in joint_ids])

    return joint_pos, joint_vel
