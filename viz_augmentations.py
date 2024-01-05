from datasets import create_dataset, create_transform
import datasets.utils as utils


dataset = create_dataset(
    config_dir="cifar10",
    config="full.yaml",
    train=True,
)

no_aug = create_transform(dataset, "no_aug.yaml")
aug = create_transform(dataset, config="rand_crop.yaml")

figsize = 1.2
for i in range(15):
    # no aug
    dataset.transform = no_aug
    img, label = dataset[i]
    cat = dataset.classes[label]
    utils.plot_img(
        img, dataset.mean, dataset.std,
        title=f"{cat} - no aug",
        figsize=figsize,
    )

    # aug
    dataset.transform = aug
    img, _ = dataset[i]
    utils.plot_img(
        img, dataset.mean, dataset.std,
        title=f"{cat} - aug",
        figsize=figsize,
    )