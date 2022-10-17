# python imports
from __future__ import print_function, absolute_import, division
import imp
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from cityscapesscripts.helpers.csHelpers import printError
import os
import glob
import sys
import time

# cityscapes imports
os.environ['CITYSCAPES_DATASET'] = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/getFine_trainvaltest"

# The main method


def main():
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
        print(cityscapesPath)
    else:
        cityscapesPath = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')
    # how to search for all ground truth
    searchFine = os.path.join(cityscapesPath, "gtFine",
                              "*", "*", "*_gt*_polygons.json")
    searchCoarse = os.path.join(
        cityscapesPath, "gtCoarse", "*", "*", "*_gt*_polygons.json")

    # search files
    filesFine = glob.glob(searchFine)
    filesFine.sort()
    filesCoarse = glob.glob(searchCoarse)
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError("Did not find any files. Please consult the README.")

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    start_time = time.time()
    for f in files:
        # create the output filename
        dst = f.replace("_polygons.json", "_labelTrainIds.png")

        # do the conversion
        try:
            json2labelImg(f, dst, "trainIds")
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        end_time = time.time()
        last_time = end_time - start_time
        print("\rProgress: {:>3} % ".format(
            progress * 100 / len(files)), end=' ')
        print("Time: {:>3} s".format(last_time), end=' ')
        sys.stdout.flush()
    print("\n")


# call the main
if __name__ == "__main__":
    main()
