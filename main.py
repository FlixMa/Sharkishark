#!/usr/local/bin/python3

import sys
import core
import argparse
from reward_driven_client import RewardDrivenClient


def main():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--host", type=str, help="The host to connect to",
                        default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, help="The port to connect to",
                        default=13050)
    parser.add_argument("-r", "--reservation", type=str, help="The reservation code")
    parser.add_argument("--opponent", action='store_true', help="flag indicating whether an opponent is automatically started")
    args = parser.parse_args()
    print('Configuration:', args)
    client = RewardDrivenClient(args.host, args.port, args.reservation)
    client.make_new_game(start_opponent=args.opponent)


if __name__ == '__main__':
    print('Arguments:', sys.argv)
    main()
