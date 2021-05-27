import os
import re


def get_best_score(f):
    lines = f.readlines()
    best_score = 0
    regex = re.compile("(?<=val_dice\=)-\d+(\.\d+)?")
    for l in lines:
        result = regex.search(l)
        if result:
            score = float(l[result.start():result.end()])
            if score < best_score:
                best_score = score
    return best_score


for d in sorted(os.listdir()):
    if not os.path.isdir(d):
        continue
    try:
        f = open(os.path.join(d, "val_log.txt"), 'r')
        best_score = get_best_score(f)
        print("{} : {}".format(d, best_score))
    except:
        print("{} : None".format(d))
