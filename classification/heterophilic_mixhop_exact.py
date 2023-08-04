from experiments.run import run
from experiments.parser import get_parser

parser = get_parser()
args = parser.parse_args()

args.dataset_name = "MixHopSynthetic"
args.model = "sage"
args.filling_method = "mixhop_exact_restriction"
args.missing_rate = 0.9
args.homophily = 0.1
args.n_runs = 3
args.num_iterations = 50

run(args)
