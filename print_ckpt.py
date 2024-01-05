import argparse
import datetime
import logger
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("Print checkpoint configuration", add_help=False)
    # If ckpt_name is a run name, load the last ckpt of the run.
    parser.add_argument(
        "--ckpt_name",
        default="test1/ckpt-10.pt",
        type=str,
    )
    return parser


def print_ckpt_info(ckpt_name):
    ckpt, ckpt_path = logger.Run.load_ckpt(ckpt_name, map_location='cpu')
    ckpt = ckpt['logger']

    # Path
    print(f"-\nPath: {ckpt_path}")

    # Date
    date = datetime.datetime(*ckpt['date'])
    date_str = date.ctime()
    print(f"-\nDate: {date_str}")

    # World size
    n_procs = ckpt['world_size']
    print(f"-\nNb processes: {n_procs}")

    # Config
    args_str = utils.args_str(ckpt['args'])
    print(f"-\n{args_str}")

    # Metrics
    mtr_lines = []
    for name in ['train', 'eval']:
        metrics = logger.MetricsLogger(name, use_window=name == 'train')
        metrics.load_state_dict(ckpt[name]['metrics'])
        mtr_str = metrics.get_str()
        if mtr_str: mtr_lines.append(mtr_str)
        if ckpt[name]['last_metrics']:
            last_metrics = logger.SummaryMetricsLogger(name)
            last_metrics.load_state_dict(ckpt[name]['last_metrics'])
            last_mtr_str = last_metrics.get_str()
            if last_mtr_str: mtr_lines.append(last_mtr_str)
        if ckpt[name]['best_metrics']:
            best_metrics = logger.SummaryMetricsLogger(name)
            best_metrics.load_state_dict(ckpt[name]['best_metrics'])
            best_mtr_str = best_metrics.get_str()
            if best_mtr_str: mtr_lines.append(best_mtr_str)
    mtr_str = utils.join_head_body("Metrics", "\n".join(mtr_lines))
    print(f"-\n{mtr_str}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    print_ckpt_info(args.ckpt_name)