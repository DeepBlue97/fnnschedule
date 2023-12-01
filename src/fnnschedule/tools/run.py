import argparse
import importlib

from fnnschedule.runner.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run args')

    parser.add_argument('mode', type=str, choices=['train', 'test', 'quant', 'deploy'], help='choose mode')
    parser.add_argument('config', type=str, help='config file')

    args = parser.parse_args()

    config = importlib.import_module(args.config)

    trainer = Trainer(config)


    # match args.mode:
    #     case 'train':
    #         # print(config.dataset)
    #         trainer.train()
    #         # trainer.model.get_model(config.model['type'], **config.model.pop('type'))
    #         # print(trainer.model.model)

    #     case 'test':
    #         trainer.test()

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()
    elif args.mode == 'deploy':
        # trainer.model.quantize_deploy(None, deploy=True)
        trainer.deploy()
