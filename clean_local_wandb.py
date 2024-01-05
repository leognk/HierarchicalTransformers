import os
import path
import shutil
import re
import wandb
from logger import WandB


def main():
    """
    Delete the runs that do not exist anymore in the WandB server
    from the local WandB directory.
    """
    if not path.Path(WandB.LOCAL_ROOT).exists():
        print(f'Path "{WandB.LOCAL_ROOT}" does not exist.')
        return
    api = wandb.Api()
    for entry in (os.listdir(WandB.LOCAL_ROOT)):
        m = re.findall(r'^run-\d+_\d+-(.+)$', entry)
        if not m: continue
        run_id = m[0]
        try:
            api.run(f"{WandB.PROJECT}/{run_id}")
        except wandb.CommError:
            shutil.rmtree(os.path.join(WandB.LOCAL_ROOT, entry))
            print(f"Removed: {entry}")
    print("Cleaning finished")


if __name__ == "__main__":
    main()