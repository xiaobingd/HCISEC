"""Reuse the previous original-trainb-core trainer without optimizer/loss changes."""
import sys,argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.baseline));import train_mei_baseline_auth as training
    # Only protocol descriptions change. The training loop is executed verbatim.
    source=Path(training.__file__).read_text()
    source=source.replace('2*largestclass=240items','2*largestclass=960items;480unique negatives and30unique positives;60updates/epoch vs15 in prior ke-only run')
    source=source.replace('Ke identity seen inTRAIN; notunseen-useropen-set.','Ke and all three Data0720 identities seen inTRAIN; notunseen-useropen-set. Prior Data0720 diagnostic already scored all sources; current recording split locked before refitting.')
    sys.argv=[str(training.__file__),'--root',str(args.root)]
    exec(compile(source,str(training.__file__),'exec'),dict(__name__='__main__',__file__=str(training.__file__)))
if __name__=='__main__':main()
