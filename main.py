import core
import argparse
from reward_driven_client import RewardDrivenClient


def main():

		parser = argparse.ArgumentParser()
		parser.add_argument("--host", type=str, help="The host to connect to",
							default="127.0.0.1")
		parser.add_argument("--port", type=int, help="The port to connect to",
							default=13050)
		parser.add_argument("--reservation", type=str, help="The reservation code")
		parser.add_argument("--opponent", action='store_true',
							help="flag indicating whether an opponent is automatically started")
		args = parser.parse_args()
		client = RewardDrivenClient(args.host, args.port, args.reservation)
		client.make_new_game(start_opponent=args.opponent)


if __name__ == '__main__':
	main()
