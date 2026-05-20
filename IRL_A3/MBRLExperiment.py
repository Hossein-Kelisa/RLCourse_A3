#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
import time
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


def run_repetitions(agent_type, n_planning_updates, wind_proportion,
                    n_timesteps, eval_interval, n_repetitions,
                    gamma, learning_rate, epsilon):

    n_eval_points = n_timesteps // eval_interval
    all_returns = np.zeros((n_repetitions, n_eval_points))

    for rep in range(n_repetitions):
        # New environment for each repetition (start from scratch)
        env = WindyGridworld(wind_proportion=wind_proportion)
        eval_env = WindyGridworld(wind_proportion=wind_proportion)

        # Choose agent
        if agent_type == 'dyna':
            pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif agent_type == 'ps':
            pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)

        s = env.reset()
        eval_idx = 0

        for t in range(n_timesteps):
            # Evaluate every eval_interval steps
            if t % eval_interval == 0 and eval_idx < n_eval_points:
                mean_return = pi.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100)
                all_returns[rep, eval_idx] = mean_return
                eval_idx += 1

            # One real step
            a = pi.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            pi.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

            if done:
                s = env.reset()
            else:
                s = s_next

        print(f"  rep {rep+1}/{n_repetitions} done")

    # Average over repetitions
    return all_returns.mean(axis=0)


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    smoothing_window = 9

    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]

    x = np.arange(n_timesteps // eval_interval) * eval_interval

    # --- Experiment 1: Dyna ---
    for wind_prop in wind_proportions:
        label = "Deterministic" if wind_prop == 1.0 else "Stochastic"
        print(f"\nDyna - {label} environment")

        plot = LearningCurvePlot(title=f"Dyna - {label} (wind={wind_prop})")

        # Q-learning baseline (n_planning=0)
        print(f" Q-learning baseline (n_plan=0)...")
        curve = run_repetitions('dyna', 0, wind_prop,
                                n_timesteps, eval_interval, n_repetitions,
                                gamma, learning_rate, epsilon)
        plot.add_curve(x, smooth(curve, window=smoothing_window), label='Q-learning (n_plan=0)')

        # Dyna with different planning budgets
        for n_plan in n_planning_updatess:
            print(f" n_planning={n_plan}...")
            curve = run_repetitions('dyna', n_plan, wind_prop,
                                    n_timesteps, eval_interval, n_repetitions,
                                    gamma, learning_rate, epsilon)
            plot.add_curve(x, smooth(curve, window=smoothing_window), label=f'Dyna (n_plan={n_plan})')

        plot.save(f'dyna_{label.lower()}.png')
        print(f"Saved: dyna_{label.lower()}.png")

    # --- Experiment 2: Prioritized Sweeping ---
    for wind_prop in wind_proportions:
        label = "Deterministic" if wind_prop == 1.0 else "Stochastic"
        print(f"\nPS - {label} environment")

        plot = LearningCurvePlot(title=f"Prioritized Sweeping - {label} (wind={wind_prop})")

        # Q-learning baseline (n_planning=0)
        print(f" Q-learning baseline (n_plan=0)...")
        curve = run_repetitions('dyna', 0, wind_prop,
                                n_timesteps, eval_interval, n_repetitions,
                                gamma, learning_rate, epsilon)
        plot.add_curve(x, smooth(curve, window=smoothing_window), label='Q-learning (n_plan=0)')

        # PS with different planning budgets
        for n_plan in n_planning_updatess:
            print(f" n_planning={n_plan}...")
            curve = run_repetitions('ps', n_plan, wind_prop,
                                    n_timesteps, eval_interval, n_repetitions,
                                    gamma, learning_rate, epsilon)
            plot.add_curve(x, smooth(curve, window=smoothing_window), label=f'PS (n_plan={n_plan})')

        plot.save(f'ps_{label.lower()}.png')
        print(f"Saved: ps_{label.lower()}.png")

    # --- Experiment 3: Comparison ---
    for wind_prop in wind_proportions:
        label = "Deterministic" if wind_prop == 1.0 else "Stochastic"
        print(f"\nComparison - {label} environment")

        plot = LearningCurvePlot(title=f"Comparison - {label} (wind={wind_prop})")

        # Q-learning baseline
        print(" Q-learning...")
        curve = run_repetitions('dyna', 0, wind_prop,
                                n_timesteps, eval_interval, n_repetitions,
                                gamma, learning_rate, epsilon)
        plot.add_curve(x, smooth(curve, window=smoothing_window), label='Q-learning (n_plan=0)')

        # Best Dyna
        print(" Best Dyna (n_plan=5)...")
        curve = run_repetitions('dyna', 5, wind_prop,
                                n_timesteps, eval_interval, n_repetitions,
                                gamma, learning_rate, epsilon)
        plot.add_curve(x, smooth(curve, window=smoothing_window), label='Dyna (n_plan=5)')

        # Best PS
        print(" Best PS (n_plan=5)...")
        curve = run_repetitions('ps', 5, wind_prop,
                                n_timesteps, eval_interval, n_repetitions,
                                gamma, learning_rate, epsilon)
        plot.add_curve(x, smooth(curve, window=smoothing_window), label='PS (n_plan=5)')

        plot.save(f'comparison_{label.lower()}.png')
        print(f"Saved: comparison_{label.lower()}.png")

    # --- Runtime Table ---
    print("\nMeasuring runtimes...")
    print(f"\n{'Algorithm':<20} {'Stochastic (s)':>15} {'Deterministic (s)':>18}")
    print("-" * 55)

    for agent_type, name, n_plan in [('dyna', 'Q-learning', 0),
                                      ('dyna', 'Dyna', 5),
                                      ('ps', 'PS', 5)]:
        times = []
        for wind_prop in wind_proportions:
            label = "Deterministic" if wind_prop == 1.0 else "Stochastic"
            print(f" Runtime {name} - {label}...")
            t_start = time.time()
            run_repetitions(agent_type, n_plan, wind_prop,
                            n_timesteps, eval_interval, 5,
                            gamma, learning_rate, epsilon)
            t_end = time.time()
            times.append((t_end - t_start) / 5)

        print(f"{name:<20} {times[0]:>15.2f} {times[1]:>18.2f}")

    print("\nAll experiments done!")


if __name__ == '__main__':
    experiment()
