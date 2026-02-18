#!/usr/bin/env python3

import argparse
import json

MOVIES_JSON_PATH = "./data/movies.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open(MOVIES_JSON_PATH, 'r') as moviesFile:
        moviesJSON = json.load(moviesFile)

    match args.command:
        case "search":
            print("Searching for:", args.query)
            searchJSON(moviesJSON, args.query)

        case _:
            parser.print_help()

def searchJSON(json, searchString) -> None:
    movieCount = 1
    for movie in json["movies"]:
        if searchString in movie["title"]:
            print("% s. % s" % (movieCount, movie["title"]))
            movieCount += 1
        if movieCount > 5:
            return


if __name__ == "__main__":
    main()