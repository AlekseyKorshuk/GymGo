import argparse

import gym

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--randai', action='store_true')
parser.add_argument('--boardsize', type=int, default=7)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi)

# Game loop
done = False
while not done:
    action = go_env.render(mode="human")
    if action == -1:
        exit()
    try:
        _, _, done, _ = go_env.step(action)
    except Exception as e:
        print(e)
        continue
    if args.randai:
        if go_env.game_ended():
            break
        action = go_env.uniform_random_action()
        _, _, done, _ = go_env.step(action)
go_env.render(mode="human")
