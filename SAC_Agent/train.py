from SAC_Agent.SAC_agent import SAC_Agent
from SAC_Agent.utils import setup_run_logging, save_scene_snapshot
from pathlib import Path
from termcolor import colored
import torch

N_EPISODES = 1000
STEPS_PER_EPISODE = 50
SAVE_WEIGHTS = True


def train():
    log_path = setup_run_logging()
    run_tag = Path(log_path).stem
    for rand_seed in [999]:
        for _ in [0.0003]:
            agent = SAC_Agent(
                seed=rand_seed,
                n_episodes=N_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                number_accumulations_before_update=1,
                max_possible_samples=256,
                memory_size=50000,
                her_ratio=0.8,
            )
            scene_captured = False
            agent.actor_optimizer.zero_grad()
            agent.critic1_optimizer.zero_grad()
            agent.critic2_optimizer.zero_grad()

            for episode in range(1, N_EPISODES + 1):
                agent.start_new_episode()
                state = agent.env.reset()
                if not scene_captured:
                    save_scene_snapshot(
                        agent.env.controller,
                        run_tag,
                        cameras=["top_down_wide", "side"],
                        width=640,
                        height=480,
                    )
                    scene_captured = True
                state_obs = agent.transform_observation(state)
                print(
                    colored(
                        f"EPISODE {episode}/{N_EPISODES}",
                        color="white",
                        attrs=["bold"],
                    )
                )

                for step in range(1, STEPS_PER_EPISODE + 1):
                    print("#################################################################")
                    print(
                        colored(
                            "EPISODE {} STEP {}".format(episode, step),
                            color="white",
                            attrs=["bold"],
                        )
                    )
                    print("#################################################################")

                    # Select action using SAC policy
                    action, _ = agent.select_action(state_obs, deterministic=False)
                    env_action = agent.transform_action(action)
                    agent.last_action = "policy"

                    next_state, reward, done, _ = agent.env.unwrapped.step(
                        env_action, record_grasps=False, action_info=agent.last_action
                    )
                    agent.update_tensorboard(reward, env_action)
                    reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
                    done_tensor = torch.tensor([[done]], dtype=torch.float32)
                    next_state_obs = agent.transform_observation(next_state)

                    agent.store_transition(
                        state_obs,
                        action,
                        next_state_obs,
                        reward_tensor,
                        done_tensor,
                        state,
                        next_state,
                    )

                    state_obs = next_state_obs
                    state = next_state
                    agent.learn()

                    if done:
                        print(
                            colored(
                                f"Episode finished early at step {step}",
                                color="yellow",
                            )
                        )
                        break

                agent.finalize_episode()

            if SAVE_WEIGHTS:
                checkpoint = {
                    "step": agent.steps_done,
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic1_state_dict": agent.critic1.state_dict(),
                    "critic2_state_dict": agent.critic2.state_dict(),
                    "critic1_target_state_dict": agent.critic1_target.state_dict(),
                    "critic2_target_state_dict": agent.critic2_target.state_dict(),
                    "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                    "critic1_optimizer_state_dict": agent.critic1_optimizer.state_dict(),
                    "critic2_optimizer_state_dict": agent.critic2_optimizer.state_dict(),
                    "greedy_rotations": agent.greedy_rotations,
                    "greedy_rotations_successes": agent.greedy_rotations_successes,
                    "random_rotations_successes": agent.random_rotations_successes,
                }
                # Save alpha-related parameters if auto_alpha is enabled
                if agent.auto_alpha:
                    checkpoint["alpha_optimizer_state_dict"] = agent.alpha_optimizer.state_dict()
                    checkpoint["log_alpha"] = agent.log_alpha.data
                    checkpoint["target_entropy"] = agent.target_entropy
                torch.save(checkpoint, agent.WEIGHT_PATH)
                # Display relative path from project root
                from SAC_Agent.SAC_agent import PROJECT_ROOT

                try:
                    relative_path = agent.WEIGHT_PATH.relative_to(PROJECT_ROOT)
                    print("Saved checkpoint to {}.".format(relative_path))
                except ValueError:
                    # Fallback to absolute path if relative conversion fails
                    print("Saved checkpoint to {}.".format(agent.WEIGHT_PATH))

            print(f"Finished training (rand_seed = {rand_seed}).")
            agent.writer.close()
            agent.env.close()


if __name__ == "__main__":
    train()
