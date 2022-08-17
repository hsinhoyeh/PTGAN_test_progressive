import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from filepath import modelpath_join

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
    scores = []
    for i in range(0, topN):
      path = inputcsvTNumTopN.index[i]
      score = inputcsvTNumTopN[i]
      # replace path
      static_path = path.replace('../AIC21/veri_pose', modelpath_join('gallery/veri_pose'))
      ims.append(Image.open(static_path))
      scores.append(score)

    savegrid(ims, scores, output, 4, 4, True, False)

def savegrid(ims, scores, save_path:str, rows=None, cols=None, fill=True, showax=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    aspratio_w, aspratio_h = 16, 9 # display photo ratio
    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    plt.style.use('dark_background')
    figsize = plt.figaspect(float(aspratio_h* rows) / float(aspratio_w * cols))
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=figsize)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        w, h = im.size
        preserved_h = w * 9 // 16
        preserved_w  = h * 16 // 9
        if preserved_h > h:
            delta_width = 0
            delta_height = preserved_h - h
        else:
            delta_width = preserved_w - w
            delta_height = 0

        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)

        im_padded = ImageOps.expand(im, padding)
        p_w, p_h = im_padded.size

        print("orig:{0}, {1}, padding:{2}, {3}".format(w,h, p_w, p_h))
        #fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)


        ax.imshow(im_padded, cmap="gray")
        ax.axis("off")
        #if not showax:
        #    ax.set_axis_off()

    for ax,score in zip(axarr.ravel(), scores):
        ax.text(0.6, 0.1, "{:0.4f}".format(score), color='white', fontsize=10,
                transform=ax.transAxes, bbox=dict(facecolor='green', alpha=0.9))
        ax.set_zorder(10)

    kwargs = {'pad_inches': .01} if fill else {}
    fig.savefig(save_path, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_matrics', dest='inputcsvmatrics', type=str, default='')
    parser.add_argument('--output', dest='output', type=str, default='')
    parser.add_argument('--top', dest='topN', type=int, default=16)

    args = parser.parse_args()
    draw_ranked_photo(args.inputcsvmatrics, args.output, args.topN)
