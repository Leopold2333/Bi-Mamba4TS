import argparse
import os
import pandas as pd


def generate_train_val_test(args):
    filepath = os.path.join(args.root_path, args.output_dir, args.traffic_df_filename+'.h5')
    df = pd.read_hdf(filepath)
    df['date'] = df.index.values
    data = df[['date', *df.columns[:-1]]]
    filepath = os.path.join(args.root_path, args.output_dir, args.traffic_df_filename+'.csv')
    data.to_csv(filepath, index=False)


def main(args):
    print("Generating training data: " + args.traffic_df_filename)
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="D:/Workspace_Python/GTformer/data", help="ROOT PATH")
    parser.add_argument("--output_dir", type=str, default="METR-LA", help="Output directory")

    parser.add_argument("--traffic_df_filename", type=str, default="metr-la", help="Raw traffic readings")
    args = parser.parse_args()
    main(args)
