import argparse

def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data", type = str, default = "assist0910", help = "dataset name"
    )
    parser.add_argument(
        "--max_len", type = int, default = 20, help = "max len"
    )
    return parser.parse_args()

