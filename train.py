"""
Utility script to launch training jobs.
"""

import argparse
import json

from leadopt.model_conf import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help='location to save the model')
    parser.add_argument('--wandb_project', help='set this to log run to wandb')
    parser.add_argument('--configuration', help='path to a configuration args.json file')

    subparsers = parser.add_subparsers(dest='version')

    for m in MODELS:
        sub = subparsers.add_parser(m)
        MODELS[m].setup_parser(sub)

    args = parser.parse_args()
    args_dict = args.__dict__

    if args.configuration is None and args.version is None:
        parser.print_help()
        exit(0)
    elif args.configuration is not None:
        _args = {}
        try:
            _args = json.loads(open(args.configuration, 'r').read())
        except Exception as e:
            print('Error reading configuration file: %s' % args.configuration)
            print(e)
            exit(-1)
        args_dict.update(_args)
    elif args.version is not None:
        pass
    else:
        print('You can specify a model or configuration file but not both.')
        exit(-1)

    # initialize model
    model_type = args_dict['version']
    model = MODELS[model_type](args_dict)

    model.train(args.save_path)


if __name__=='__main__':
    main()
