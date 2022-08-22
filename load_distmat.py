import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def draw_ranked_photo(csvfile: str, output: str, topN: int):
    inputcsv = pd.read_csv(csvfile)
    # do transportT such that each row represents one gallery path with one distance
    inputcsvT = inputcsv.T

    # skip the first row as it displays query information
    inputcsvT = inputcsvT[1:]

    # convert the 0th column type to number
    inputcsvTNum = pd.to_numeric(inputcsvT[0])
    inputcsvTNumTopN = inputcsvTNum.nsmallest(topN)
    ims = []
    for i in range(0, topN):
      path = inputcsvTNumTopN.index[i]
      score = inputcsvTNumTopN[i]
      # replace path
      static_path = path.replace('../AIC21/veri_pose', '/models/gallery/veri_pose')
      ims.append(Image.open(static_path))

    savegrid(ims, output, 4, 4, True, False)

def savegrid(ims, save_path:str, rows=None, cols=None, fill=True, showax=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        ax.imshow(im)
        if not showax:
            ax.set_axis_off()

    kwargs = {'pad_inches': .01} if fill else {}
    fig.savefig(save_path, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_matrics', dest='inputcsvmatrics', type=str, default='')
    parser.add_argument('--output', dest='output', type=str, default='')
    parser.add_argument('--top', dest='topN', type=int, default=16)

    args = parser.parse_args()
    draw_ranked_photo(args.inputcsvmatrics, args.output, args.topN)
