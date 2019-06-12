#!/usr/local/bin/python3
import sys
import argparse
import socket
import time

from main import main as run

def main():

    parser = argparse.ArgumentParser(description='Even Simpler Client - Team SharkiShark')
    parser.add_argument('--host', type=str, help='The host the client should connect to.', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='The port the client should connect to.', default=13050)
    parser.add_argument('--reservation', type=str, help='The reservation code of the game it should enter.', default=None)

    args = parser.parse_args()

    with open('/Users/Felix/Documents/Entwickler/PWB_2019/reinforcement_learning/log/args.txt', 'w') as file:
        file.write("sys.argv: %s\n" % sys.argv)

        file.write("\n\n parser:\n")
        file.write("\thost: %s (%s)\n" % (repr(args.host), type(args.host)))
        file.write("\tport: %s (%s)\n" % (repr(args.port), type(args.port)))
        file.write("\treservation code: %s (%s)\n" % (repr(args.reservation), type(args.reservation)))

    run(args.host, args.port)



if __name__ == '__main__':
    main()
