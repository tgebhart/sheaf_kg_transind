from experiments.run import run
from experiments.parser import get_parser

parser = get_parser()
args = parser.parse_args()

args.dataset_name = 'MixHopSynthetic'
args.model = 'sage'
args.filling_method = 'constant_propagation'
args.missing_rate = 0.9
args.homophily = 0.9
args.n_runs = 3
args.num_iterations = 50

run(args)