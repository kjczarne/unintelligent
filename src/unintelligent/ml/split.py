from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Iterable
import os
from rich.table import Table
from rich.console import Console
from abc import abstractmethod
import shutil


@dataclass
class LabeledDatasetSplitter:
    """Splits images into labeled subfolders. It uses a Pandas DataFrame
    to find the image paths and labels and the splitting is done at
    the level of the DataFrame initially.
    """
    df: pd.DataFrame
    label_series_name: str
    images_directory: Path
    output_directory: Path
    img_ext: str = ".png"

    @abstractmethod
    def find_image(self, row: pd.Series) -> Path:
        """Finds the image path using unique image ID from
        a given dataset.
        """
        raise NotImplementedError("Not implemented")

    def copy_image(self, img_input_path: Path, slice_dir_path: Path):
        """Copies image from a specified input path
        to the output directory specified at instantiation of the
        `DatasetSplitter` object.
        """
        shutil.copy(img_input_path, slice_dir_path)

    def split(self,
              splits: Iterable[str] = ("train", "test", "val"),
              proportions: Iterable[float] = (0.8, 0.1, 0.1),
              seed: int = 69,
              dry_run: bool = False,
              debug: bool = False):
        """Splits the dataset into `splits`"""

        if len(splits) != len(proportions):
            raise ValueError(
                f"Splits and proportions are of different lengths: {len(splits)} vs. {len(proportions)}"
            )

        # Shuffle rows in the data frame randomly
        shuffled_df = self.df.sample(frac=1, random_state=seed)
        shuffled_dfs_for_each_class = []
        slices = []

        for split, proportion in zip(splits, proportions):
            new_df = shuffled_df.iloc[0:0]  # empty df with the same columns
            for class_df_idx in shuffled_df[self.label_series_name].unique():
                begin_idx = 0
                class_df = shuffled_df[shuffled_df[self.label_series_name] == class_df_idx]
                class_df = class_df.reset_index()
                # choosing a split proportion _of each of the classes_
                nrow = round(class_df.shape[0] * proportion)
                data_slice = class_df.iloc[begin_idx:begin_idx + nrow + 1]
                begin_idx += nrow + 1
                new_df = pd.concat((new_df, data_slice))
            slices.append(new_df)

            if not dry_run:
                split_dir = self.output_directory / split
                if not split_dir.exists():
                    os.makedirs(split_dir)
                new_df.to_csv(split_dir / "metadata.csv")

        if not dry_run:
            for split, slice in zip(splits, slices):
                print(f"Handling {split}")
                for _, row in tqdm(slice.iterrows(), position=0, leave=True):
                    img_path = self.find_image(row)
                    self.copy_image(img_path, self.output_directory / split)

        if debug:
            actual_counts = [i.shape[0] for i in slices]
            table1 = Table(title="Summary")

            table1.add_column("Description", justify="left")
            table1.add_column("Value", justify="right", style="cyan")

            table1.add_row("Total in dataset:", str(self.df.shape[0]))
            table1.add_row("Sum of slice counts:", str(sum(actual_counts)))
            
            table2 = Table(title="Slice shapes")
            table2.add_column("Split", justify="left")
            table2.add_column("Shape", justify="right", style="cyan")
            for split, s in zip(splits, slices):
                table2.add_row(split, str(s.shape))

            table3 = Table(title="Proportion debug table")
            table3.add_column("Split", justify="left")
            table3.add_column("Expected Proportion", justify="right", style="red")
            table3.add_column("Actual Proportion", justify="right", style="cyan")

            for split, proportion, actual_num in zip(splits, proportions, actual_counts):
                actual_proportion = actual_num / sum(actual_counts)
                table3.add_row(split, str(proportion), str(actual_proportion))

            table4 = Table(title="Class Proportions for each split")
            table4.add_column("Split", justify="left")
            table4.add_column("Number of samples", justify="right", style="cyan")
            
            for split, s in zip(splits, slices):
                for class_df_idx in s[self.label_series_name].unique():
                    sample_count_for_class = s[s[self.label_series_name] == class_df_idx][self.label_series_name].count()
                    table4.add_row(f"Samples for class {class_df_idx} ({split})", str(sample_count_for_class))

            console = Console()
            console.print(table1)
            console.print(table2)
            console.print(table3)
            console.print(table4)


@dataclass
class GenericLabeledDatasetSplitter(LabeledDatasetSplitter):
    """Assumes that the rows in the dataframe contain the
    bare file image names without the extension in the `img_name` column."""

    def find_image(self, row: pd.Series) -> Path:
        return self.images_directory / (row['img_name'] + self.img_ext)


@dataclass
class UnlabeledDatasetSplitter:
    pass
    # TODO: unlabeled splitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset",
                        choices=["labeled", "unlabeled"],
                        help="Dataset to split",
                        required=True)
    parser.add_argument('-i', "--images-dir",
                        type=str,
                        help="Path to the directory containing dataset images",
                        required=True)
    parser.add_argument('-m', "--metadata-file",
                        type=str,
                        help="Path to the dataset metadata file (post-filtering, corrected metadata)",
                        required=True)
    parser.add_argument('-l', "--label-col",
                        type=str,
                        help="Label column name",
                        required=True)
    parser.add_argument('-s', "--splits",
                        nargs='+',
                        default=('train', 'test', 'val'),
                        help="Dataset splits to perform")
    parser.add_argument('-o', "--out-dir",
                        type=str,
                        help="Path to the output directory for images." \
                             "`--splits`-determined subfolders will be writen to said directory",
                        required=True)
    parser.add_argument('-p', "--proportions",
                        nargs="+",
                        type=float,
                        default=(0.8, 0.1, 0.1),
                        help="Split proportions to use, same order as `--splits`")
    parser.add_argument("--seed", type=int, help="Random integer seed", default=69)
    parser.add_argument("--dry-run", action="store_true", help="Dry run, does not copy images")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Prints detailed information about the splitting process")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_file, header=0)

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    match args.dataset.lower():
        case "labeled":
            ds = GenericLabeledDatasetSplitter(df, args.label_col, images_dir, out_dir)
        case _:
            raise ValueError(f"Dataset '{args.dataset}' is not supported splitter")

    ds.split(args.splits, args.proportions, args.seed, args.dry_run, args.debug)


if __name__ == "__main__":
    main()
