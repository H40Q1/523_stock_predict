from argparse import ArgumentParser

def get_arguments():
    """Defines command-line arguments, and parses them.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=100,
        help="The hidden size. Default: 100")

    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="The time step. Default: 20")

    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="The number of LSTM layer. Default: 2")

    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-5,
        help="The learning rate. Default: 5e-5")


    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs. Default: 200")

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="The batch size. Default: 64")

    parser.add_argument(
        "--stock-path",
        type=str,
        default="./data/AAPL.csv",
        help="Path to the root directory of the selected dataset. "
        "Default: ./data/AAPL.csv")

    parser.add_argument(
        "--stock-name",
        type=str,
        default="AAPL",
        help="Name of stock "
             "Default: AAPL")

    return parser.parse_args()


