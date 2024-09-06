import os
from pytorch_lightning.cli import LightningCLI

class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--experiment', type = str, required = True,
                            help = 'Name of experiment')
        parser.add_argument('--checkpoint', type = str, default = 'checkpoints',
                            help = 'Path where checkpoints are to be saved')
        parser.add_argument('--logs', type = str, default = 'logs',
                            help = 'Path where logs are to be saved')
        parser.link_arguments('data.seq_length', 'model.init_args.model.seq_length')
        # parser.link_arguments('data.vocab', 'model.init_args.vocab', apply_on = 'instantiate')
        # parser.link_arguments('data.vocab_size', 'model.init_args.model.vocab_size', apply_on = 'instantiate')

    def before_instantiate_classes(self):
        subcommand = self.config.subcommand
        log = self.config[subcommand].logs
        experiment = self.config[subcommand].experiment
        checkpoint = self.config[subcommand].checkpoint
        self.config[subcommand].trainer.default_root_dir = os.path.join(log, experiment)
        for callback in self.config[subcommand].trainer.callbacks:
            if 'dirpath' in callback.init_args:
                callback.init_args.dirpath = os.path.join(checkpoint, experiment)
        
        if subcommand in ['validate', 'test']:
            assert self.config[subcommand].ckpt_path is not None, 'ckpt_path needs to be provided'
            self.config[subcommand].ckpt_path = os.path.join(checkpoint,
                                                             experiment,
                                                             self.config[subcommand].ckpt_path)

if __name__ == '__main__':
    cli = MyLightningCLI()