import sys
import argparse
import os
import pickle
# Torch imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

# Homemade modules
from train.loader import Loader, loader_from_hparams
from train.model import Model, model_from_hparams
from train.learn import train_model
from tools.learning_utils import mkdirs_learning, ConfParser

def main():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("-da", "--interaction_type", default='rna')
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="choose the batch size")
    parser.add_argument("-nw", "--workers", type=int, default=0, help="Number of workers to load data")
    parser.add_argument("-wt", "--wall_time", type=int, default=None, help="Max time to run the model")
    parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")
    parser.add_argument("-t", "--timed", help="to use timed learning", action='store_true')
    parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs to train", default=30)
    parser.add_argument("-dev", "--device", default=0, type=int, help="gpu device to use")

    # Reconstruction arguments
    parser.add_argument('--optim', type=str,
                        help='Supported Options: sgd, adam',
                        default="adam")
    parser.add_argument('-lr', '--lr', type=float,
                        default=0.01)
    parser.add_argument("-sl", "--self_loop", default=False,
                        help="Add a self loop to graphs for convolution. Default: False",
                        action='store_true'),
    parser.add_argument('-ed', '--embedding_dims', nargs='+', type=int, help='Dimensions for embeddings.',
                        default=[32, 16, 1])
    args, _ = parser.parse_known_args()

    print(f"OPTIONS USED \n ",
          '-' * 10 + '\n',
          '\n'.join(map(str, vars(args).items()))
          )

    hparams = ConfParser(argparse=args)

    # Hardware settings
    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Dataloader creation
    graphs_path = os.path.join(script_dir,
                                '../data/graphs/interfaces_cutoff10', args.interaction_type)
    loader = loader_from_hparams(graphs_path=graphs_path, hparams=hparams)
    hparams.add_value('argparse', 'num_edge_types', loader.num_edge_types)
    train_loader, test_loader, all_loader = loader.get_data()

    if len(train_loader) == 0 & len(test_loader) == 0:
        raise ValueError('there are not enough points compared to the BS')

    # Model and optimizer setup
    model = model_from_hparams(hparams=hparams)
    model = model.to(device)
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Experiment Setup
    name = args.name
    result_folder, save_path = mkdirs_learning(name)

    writer = SummaryWriter(result_folder)
    print(f'Saving result in {name}')

    # write model metadata
    hparams.dump(dump_path=os.path.join(script_dir, '../results/trained_models', args.name, f'{args.name}.exp'))
    pickle.dump({
        'dims': args.embedding_dims,
        'edge_map': train_loader.dataset.dataset.edge_map,
    },
        open(os.path.join(os.path.dirname(save_path), 'meta.p'), 'wb'))

    # Run
    train_model(model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                save_path=save_path,
                writer=writer,
                num_epochs=args.num_epochs,
                wall_time=args.wall_time)

if __name__ == '__main__':
    main()
