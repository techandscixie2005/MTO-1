"""Production interface. All substantive imports and writes occur after the exact gate."""
import argparse
from common import require_cpu
from execution_gate import verify_execution


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('fit_export','evaluate'))
    for name in ('authorization','manifest','review','publication'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    require_cpu()
    permit=verify_execution(args.authorization,args.manifest,args.review,args.publication)
    # Neither NumPy/Torch nor a model/data module is imported before authorization.
    from production_stages import run_stage
    return run_stage(args.stage,permit)


if __name__=='__main__':
    main()
