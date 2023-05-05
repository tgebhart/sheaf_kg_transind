from experiments.run import run
from experiments.parser import get_parser

parser = get_parser()
args = parser.parse_args()

args.dataset_name = 'MixHopSynthetic'
args.model = 'sage'
args.filling_method = 'constant_propagation'
args.missing_rate = 0.5
args.homophily = 0.1

run(args)