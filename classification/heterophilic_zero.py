from experiments.run import run
from experiments.parser import get_parser

parser = get_parser()
args = parser.parse_args()

args.dataset_name = "MixHopSynthetic"
args.model = "sage"
args.filling_method = "zero"
args.missing_rate = 0.99
args.homophily = 0.8
args.n_runs = 1

run(args)
