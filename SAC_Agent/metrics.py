import numpy as np
from collections import deque


class MotionQualityMetrics:
    """
    Tracks and computes motion quality metrics for robotic manipulation.
    Metrics include: jerk, RMS acceleration, vertical overshoot, smoothness, and collisions.
    """

    def __init__(self, table_height=0.91, history_size=1000):
        """
        Initialize metrics tracker.

        Args:
            table_height: Height of the table surface (z-coordinate)
            history_size: Maximum number of timesteps to store in history
        """
        self.table_height = table_height
        self.history_size = history_size

        # History buffers
        self.joint_positions = deque(maxlen=history_size)
        self.joint_velocities = deque(maxlen=history_size)
        self.joint_accelerations = deque(maxlen=history_size)
        self.end_effector_positions = deque(maxlen=history_size)
        self.timesteps = deque(maxlen=history_size)

        # Collision tracking
        self.table_collision_count = 0
        self.geometry_collision_count = 0
        self.last_contact_count = 0

        # Episode statistics
        self.episode_metrics = {
            'mean_jerk': 0.0,
            'rms_acceleration': 0.0,
            'vertical_overshoot': 0.0,
            'max_vertical_overshoot': 0.0,
            'smoothness_index': 0.0,
            'table_collisions': 0,
            'geometry_collisions': 0,
            'total_collisions': 0
        }

    def update(self, mujoco_data, actuated_joint_ids, ee_body_id, dt):
        """
        Update metrics with current simulation state.

        Args:
            mujoco_data: MuJoCo data object (mj.MjData)
            actuated_joint_ids: Array of actuated joint indices
            ee_body_id: End-effector body ID
            dt: Time step duration
        """
        # Store joint states
        joint_pos = mujoco_data.qpos[actuated_joint_ids].copy()
        joint_vel = mujoco_data.qvel[actuated_joint_ids].copy()
        joint_acc = mujoco_data.qacc[actuated_joint_ids].copy()

        self.joint_positions.append(joint_pos)
        self.joint_velocities.append(joint_vel)
        self.joint_accelerations.append(joint_acc)
        self.timesteps.append(dt)

        # Store end-effector position
        if ee_body_id is not None:
            ee_pos = mujoco_data.xpos[ee_body_id].copy()
            self.end_effector_positions.append(ee_pos)

        # Check for collisions
        self._update_collisions(mujoco_data)

    def _update_collisions(self, mujoco_data):
        """
        Update collision counts by checking contacts.
        """
        n_contacts = mujoco_data.ncon
        current_contacts = set()

        for i in range(n_contacts):
            contact = mujoco_data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Check if contact involves table (assuming table geom name contains "table")
            # or check by geometry properties
            # This is a simplified check - you may need to adjust based on your model
            contact_pair = (min(geom1_id, geom2_id), max(geom1_id, geom2_id))
            current_contacts.add(contact_pair)

            # Check for table collisions (heuristic: check if contact is near table height)
            contact_pos = contact.pos
            if abs(contact_pos[2] - self.table_height) < 0.05:
                # Likely table contact
                if contact_pair not in getattr(self, '_table_contacts', set()):
                    self.table_collision_count += 1
                    if not hasattr(self, '_table_contacts'):
                        self._table_contacts = set()
                    self._table_contacts.add(contact_pair)

        # Count new geometry collisions (non-table)
        if hasattr(self, '_last_contacts'):
            new_contacts = current_contacts - self._last_contacts
            for contact_pair in new_contacts:
                # Check if it's not a table contact
                if contact_pair not in getattr(self, '_table_contacts', set()):
                    self.geometry_collision_count += 1

        self._last_contacts = current_contacts

    def compute_mean_jerk(self):
        """
        Compute mean jerk (rate of change of acceleration) across all joints.

        Returns:
            Mean jerk value
        """
        if len(self.joint_accelerations) < 2:
            return 0.0

        jerks = []
        for i in range(1, len(self.joint_accelerations)):
            dt = self.timesteps[i] if i < len(self.timesteps) else self.timesteps[-1]
            if dt > 0:
                jerk = (
                    self.joint_accelerations[i] - self.joint_accelerations[i - 1]
                ) / dt
                jerks.append(np.abs(jerk))

        if len(jerks) == 0:
            return 0.0

        # Mean jerk across all joints and all timesteps
        all_jerks = np.concatenate(jerks)
        return np.mean(all_jerks)

    def compute_rms_acceleration(self):
        """
        Compute root-mean-square joint acceleration.

        Returns:
            RMS acceleration value
        """
        if len(self.joint_accelerations) == 0:
            return 0.0

        # Compute RMS across all joints and timesteps
        all_accelerations = np.array(list(self.joint_accelerations))
        rms = np.sqrt(np.mean(all_accelerations ** 2))
        return rms

    def compute_vertical_overshoot(self):
        """
        Compute vertical overshoot (end-effector going below table height).

        Returns:
            Tuple of (mean_overshoot, max_overshoot) in meters
        """
        if len(self.end_effector_positions) == 0:
            return 0.0, 0.0

        overshoots = []
        for ee_pos in self.end_effector_positions:
            if ee_pos[2] < self.table_height:
                overshoot = self.table_height - ee_pos[2]
                overshoots.append(overshoot)

        if len(overshoots) == 0:
            return 0.0, 0.0

        mean_overshoot = np.mean(overshoots)
        max_overshoot = np.max(overshoots)
        return mean_overshoot, max_overshoot

    def compute_smoothness_index(self, method='spectral_arc_length'):
        """
        Compute smoothness index using different methods.

        Args:
            method: 'spectral_arc_length' or 'log_dimensionless_jerk'

        Returns:
            Smoothness index (higher is smoother)
        """
        if len(self.joint_velocities) < 3:
            return 0.0

        if method == 'spectral_arc_length':
            return self._spectral_arc_length()
        elif method == 'log_dimensionless_jerk':
            return self._log_dimensionless_jerk()
        else:
            raise ValueError(f"Unknown smoothness method: {method}")

    def _spectral_arc_length(self):
        """
        Compute spectral arc length smoothness metric.
        Higher values indicate smoother motion.
        """
        # Use end-effector velocity if available, otherwise use joint velocities
        if len(self.end_effector_positions) < 3:
            # Fall back to joint velocities
            velocities = np.array(list(self.joint_velocities))
            # Use first joint as proxy
            vel_magnitude = np.linalg.norm(velocities[:, 0]) if velocities.shape[1] > 0 else 0
        else:
            # Compute end-effector velocity from positions
            ee_positions = np.array(list(self.end_effector_positions))
            dt = np.mean(list(self.timesteps)) if len(self.timesteps) > 0 else 0.01
            if dt > 0:
                ee_velocities = np.diff(ee_positions, axis=0) / dt
                vel_magnitude = np.linalg.norm(ee_velocities, axis=1)
            else:
                return 0.0

        if len(vel_magnitude) < 2:
            return 0.0

        # Compute FFT
        fft = np.fft.fft(vel_magnitude)
        freqs = np.fft.fftfreq(len(vel_magnitude))

        # Normalize
        fft_magnitude = np.abs(fft)
        if np.sum(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.sum(fft_magnitude)

        # Compute arc length in frequency domain
        # Use only positive frequencies
        positive_freqs = freqs > 0
        if np.sum(positive_freqs) < 2:
            return 0.0

        fft_positive = fft_magnitude[positive_freqs]
        freq_positive = freqs[positive_freqs]

        # Arc length
        df = np.diff(freq_positive)
        dfft = np.diff(fft_positive)
        arc_length = np.sum(np.sqrt(df**2 + dfft**2))

        # Smoothness is inverse of arc length (normalized)
        smoothness = 1.0 / (1.0 + arc_length)
        return smoothness

    def _log_dimensionless_jerk(self):
        """
        Compute log dimensionless jerk smoothness metric.
        Higher (less negative) values indicate smoother motion.
        """
        if len(self.joint_accelerations) < 3:
            return 0.0

        # Compute jerk
        jerks = []
        for i in range(1, len(self.joint_accelerations)):
            dt = self.timesteps[i] if i < len(self.timesteps) else self.timesteps[-1]
            if dt > 0:
                jerk = (
                    self.joint_accelerations[i] - self.joint_accelerations[i - 1]
                ) / dt
                jerks.append(jerk)

        if len(jerks) == 0:
            return 0.0

        jerks = np.array(jerks)
        # Use first joint as proxy, or average across joints
        if jerks.ndim > 1:
            jerk_magnitude = np.linalg.norm(jerks, axis=1)
        else:
            jerk_magnitude = np.abs(jerks)

        # Compute movement duration and distance
        if len(self.joint_positions) > 0:
            positions = np.array(list(self.joint_positions))
            if positions.ndim > 1 and positions.shape[1] > 0:
                # Use first joint position
                movement_distance = np.max(positions[:, 0]) - np.min(positions[:, 0])
                movement_duration = len(self.timesteps) * np.mean(list(self.timesteps))
            else:
                return 0.0
        else:
            return 0.0

        if movement_distance == 0 or movement_duration == 0:
            return 0.0

        # Dimensionless jerk
        jerk_integral = np.trapz(jerk_magnitude ** 2, dx=movement_duration / len(jerk_magnitude))
        dimensionless_jerk = -np.log(
            (movement_duration**3 / movement_distance**2) * jerk_integral + 1e-10
        )

        return dimensionless_jerk

    def get_collision_counts(self):
        """
        Get collision counts.

        Returns:
            Tuple of (table_collisions, geometry_collisions, total_collisions)
        """
        total = self.table_collision_count + self.geometry_collision_count
        return self.table_collision_count, self.geometry_collision_count, total

    def compute_all_metrics(self):
        """
        Compute all motion quality metrics.

        Returns:
            Dictionary of all computed metrics
        """
        mean_jerk = self.compute_mean_jerk()
        rms_acc = self.compute_rms_acceleration()
        mean_overshoot, max_overshoot = self.compute_vertical_overshoot()
        smoothness = self.compute_smoothness_index()
        table_coll, geom_coll, total_coll = self.get_collision_counts()

        self.episode_metrics = {
            'mean_jerk': mean_jerk,
            'rms_acceleration': rms_acc,
            'vertical_overshoot': mean_overshoot,
            'max_vertical_overshoot': max_overshoot,
            'smoothness_index': smoothness,
            'table_collisions': table_coll,
            'geometry_collisions': geom_coll,
            'total_collisions': total_coll
        }

        return self.episode_metrics.copy()

    def reset(self):
        """
        Reset metrics for a new episode.
        """
        self.joint_positions.clear()
        self.joint_velocities.clear()
        self.joint_accelerations.clear()
        self.end_effector_positions.clear()
        self.timesteps.clear()

        self.table_collision_count = 0
        self.geometry_collision_count = 0
        self.last_contact_count = 0

        if hasattr(self, '_table_contacts'):
            self._table_contacts.clear()
        if hasattr(self, '_last_contacts'):
            self._last_contacts.clear()

        self.episode_metrics = {
            'mean_jerk': 0.0,
            'rms_acceleration': 0.0,
            'vertical_overshoot': 0.0,
            'max_vertical_overshoot': 0.0,
            'smoothness_index': 0.0,
            'table_collisions': 0,
            'geometry_collisions': 0,
            'total_collisions': 0
        }
