import numpy as np


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) for goal-conditioned RL.
    Relabels failed episodes with achieved goals as desired goals.
    """

    def __init__(self, replay_buffer, strategy='future', k=4):
        """
        Initialize HER.

        Args:
            replay_buffer: The replay buffer to store transitions
            strategy: Strategy for goal relabeling ('future', 'final', 'episode')
            k: Number of additional goals to sample per transition (for 'future' strategy)
        """
        self.replay_buffer = replay_buffer
        self.strategy = strategy
        self.k = k

    def store_episode(self, episode_transitions):
        """
        Store an episode and create HER transitions.

        Args:
            episode_transitions: List of (state, action, reward, next_state, done, info) tuples
        """
        # Store original transitions
        for transition in episode_transitions:
            state, action, reward, next_state, done, _ = transition
            self.replay_buffer.push(state, action, reward, next_state, done)

        # Create HER transitions
        if self.strategy == 'future':
            self._relabel_future(episode_transitions)
        elif self.strategy == 'final':
            self._relabel_final(episode_transitions)
        elif self.strategy == 'episode':
            self._relabel_episode(episode_transitions)

    def _relabel_future(self, episode_transitions):
        """
        Relabel goals using future strategy: sample k future achieved goals.
        """
        T = len(episode_transitions)

        for t in range(T):
            state, action, _, next_state, done, info = episode_transitions[t]
            achieved_goal = info.get('achieved_goal', next_state.get('achieved_goal'))

            # Sample k future goals
            rng = np.random.default_rng()
            future_range = list(range(t + 1, T))
            if len(future_range) > 0:
                future_indices = rng.choice(
                    future_range,
                    size=min(self.k, len(future_range)),
                    replace=False
                )
            else:
                future_indices = []

            for idx in future_indices:
                _, _, _, future_next_state, _, future_info = episode_transitions[idx]
                future_achieved_goal = future_info.get('achieved_goal', future_next_state.get('achieved_goal'))
                future_grasp_success = future_info.get('grasp_success', future_info.get('is_success', 0.0))

                # Create relabeled transition
                relabeled_state = state.copy()
                relabeled_state['desired_goal'] = future_achieved_goal.copy()

                relabeled_next_state = next_state.copy()
                relabeled_next_state['desired_goal'] = future_achieved_goal.copy()

                # Compute reward with new goal (only full reward if something was grasped)
                reward = self._compute_reward(achieved_goal, future_achieved_goal, grasp_success=future_grasp_success)

                # Done is True if we reached the relabeled goal AND something was grasped
                done = self._is_success(achieved_goal, future_achieved_goal, grasp_success=future_grasp_success)

                self.replay_buffer.push(relabeled_state, action, reward, relabeled_next_state, done)

    def _relabel_final(self, episode_transitions):
        """
        Relabel goals using final strategy: use final achieved goal as desired goal.
        """
        if len(episode_transitions) == 0:
            return

        # Get final achieved goal
        final_transition = episode_transitions[-1]
        _, _, _, final_next_state, _, final_info = final_transition
        final_achieved_goal = final_info.get('achieved_goal', final_next_state.get('achieved_goal'))
        final_grasp_success = final_info.get('grasp_success', final_info.get('is_success', 0.0))

        # Relabel all transitions with final goal
        for t in range(len(episode_transitions)):
            state, action, _, next_state, done, info = episode_transitions[t]
            achieved_goal = info.get('achieved_goal', next_state.get('achieved_goal'))

            relabeled_state = state.copy()
            relabeled_state['desired_goal'] = final_achieved_goal.copy()

            relabeled_next_state = next_state.copy()
            relabeled_next_state['desired_goal'] = final_achieved_goal.copy()

            reward = self._compute_reward(achieved_goal, final_achieved_goal, grasp_success=final_grasp_success)
            done = self._is_success(achieved_goal, final_achieved_goal, grasp_success=final_grasp_success)

            self.replay_buffer.push(relabeled_state, action, reward, relabeled_next_state, done)

    def _relabel_episode(self, episode_transitions):
        """
        Relabel goals using episode strategy: sample random achieved goal from episode.
        """
        T = len(episode_transitions)
        if T == 0:
            return

        # Collect all achieved goals and their grasp success status
        achieved_goals = []
        grasp_successes = []
        for transition in episode_transitions:
            _, _, _, next_state, _, info = transition
            achieved_goal = info.get('achieved_goal', next_state.get('achieved_goal'))
            grasp_success = info.get('grasp_success', info.get('is_success', 0.0))
            achieved_goals.append(achieved_goal)
            grasp_successes.append(grasp_success)

        # Relabel each transition with a random goal from the episode
        rng = np.random.default_rng()
        for t in range(T):
            state, action, _, next_state, done, info = episode_transitions[t]
            achieved_goal = info.get('achieved_goal', next_state.get('achieved_goal'))

            # Sample random goal from episode
            random_idx = rng.integers(0, T)
            random_goal = achieved_goals[random_idx]
            random_grasp_success = grasp_successes[random_idx]

            relabeled_state = state.copy()
            relabeled_state['desired_goal'] = random_goal.copy()

            relabeled_next_state = next_state.copy()
            relabeled_next_state['desired_goal'] = random_goal.copy()

            reward = self._compute_reward(achieved_goal, random_goal, grasp_success=random_grasp_success)
            done = self._is_success(achieved_goal, random_goal, grasp_success=random_grasp_success)

            self.replay_buffer.push(relabeled_state, action, reward, relabeled_next_state, done)

    def _compute_reward(self, achieved_goal, desired_goal, goal_tolerance=0.03, grasp_success=0.0):
        """
        Compute reward based on distance between achieved and desired goal.
        Only gives full reward if position is close AND something was grasped.
        This prevents rewarding empty gripper movements to goal locations.

        Args:
            achieved_goal: The achieved goal position
            desired_goal: The desired goal position
            goal_tolerance: Distance tolerance for success
            grasp_success: Whether something was actually grasped (1.0) or not (0.0)
        """
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])

        if distance <= goal_tolerance and grasp_success > 0.5:
            # Full reward only if position is close AND something was grasped
            return 0.5
        clipped_distance = min(distance, 1.0)
        shaping = -0.25 * clipped_distance
        return shaping

    def _is_success(self, achieved_goal, desired_goal, goal_tolerance=0.03, grasp_success=0.0):
        """
        Check if achieved goal is within tolerance of desired goal.
        Only returns True if position is close AND something was grasped.
        This prevents marking empty gripper movements as successful.

        Args:
            achieved_goal: The achieved goal position
            desired_goal: The desired goal position
            goal_tolerance: Distance tolerance for success
            grasp_success: Whether something was actually grasped (1.0) or not (0.0)
        """
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        # Success requires both position proximity AND actual grasp
        return distance <= goal_tolerance and grasp_success > 0.5
