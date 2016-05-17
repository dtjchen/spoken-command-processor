import sys, os
import click
import processor
from model import speech2phonemes, phonemes2text


DEBUG = True

@click.group()
def cli():
    pass

@cli.command()
@click.option('--data', default=None, help='size of the data used to train the model')
@click.option('--summarize', default=False, help='plot the loss function')
@click.argument('model')
@click.argument('action')
def model(**kwargs):
    if kwargs['model'] == 'speech2phonemes':
        if kwargs['action'] == 'train':
            click.echo('Training speech2phonemes...')
            speech2phonemes.train(summarize=kwargs['summarize'], data_limit=kwargs['data'])

        elif kwargs['action'] == 'test':
            click.echo('Testing speech2phonemes...')
            speech2phonemes.test()

    elif kwargs['model'] == 'phonemes2text':
        if kwargs['action'] == 'train':
            click.echo('Training phonemes2text...')
            phonemes2text.train(summarize=kwargs['summarize'], data_limit=kwargs['data'])

        elif kwargs['action'] == 'test':
            click.echo('Testing phonemes2text...')
            phonemes2text.test()

    else:
        click.echo('Unrecognized model: %s' % kwargs['model'])

@cli.command()
def register(**kwargs):
    processor.register()

@cli.command()
def parse(**kwargs):
    processor.parse()


if __name__ == '__main__':
    # Suppress stderr from the output
    if not DEBUG:
        null = open(os.devnull,'wb')
        sys.stderr = null

    # Activate the command-line interface
    cli()
