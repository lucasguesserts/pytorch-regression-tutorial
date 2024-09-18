import os

import pandas as pd


def read(dir, file_name):
    file_path = os.path.join(dir, file_name)
    data = pd.read_csv(file_path)
    return data

def save(data, dir, file_name):
    output_file_path = os.path.join(dir, file_name)
    data.to_csv(output_file_path, index=False)
    return

def transform(data, x, y):
    # sort
    idx = data.groupby(by=x)[y].idxmin()
    data = data.loc[idx]
    data = data.sort_values(by=[x, y])
    # reorder columns
    data = data[[x, y] + [c for c in data.columns if c not in [x, y]]]
    return data


def plot(data, x, y, dir):
    # plot
    ax = data.plot(
        x=x,
        y=y,
        loglog=True,
        grid=True,
        xlabel=x,
        ylabel=y,
        legend=False,
        style="-",
        marker=".",
        color="black",
        title=f"{y}\nas a function of\n{x}",
    )
    # save
    output_figure_name = f"{x.replace(" ", "_")}_as_a_function_of_{y.replace(" ", "_")}.jpg"
    output_figure_path = os.path.join(dir, output_figure_name)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_figure_path)
    return


if __name__ == "__main__":
    dir = "output"
    file_name = "curve_fit.csv"
    x="number of trainable parameters"
    y="mean square error"
    output_file_name = "curve_fit_selected.csv"

    data = read(
        dir=dir,
        file_name=file_name,
    )

    data = transform(
        data=data,
        x=x,
        y=y,
    )

    plot(
        data=data,
        x=x,
        y=y,
        dir=dir,
    )

    save(
        data=data,
        dir=dir,
        file_name=output_file_name,
    )

