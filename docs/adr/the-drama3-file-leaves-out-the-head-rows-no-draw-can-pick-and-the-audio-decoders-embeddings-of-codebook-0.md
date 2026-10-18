# The Drama 3 file leaves out the head rows no draw can pick and the audio decoder's embeddings of codebook 0

## Context

Drama 3's checkpoint holds the text model's output head for its whole vocabulary, 252,288 rows of 2,048 (1.03 GB in
BF16, 2.07 GB in F32, 0.55 GB in Q8_0), but the main token of a frame is drawn from the end id and the 4,096 semantic
ids alone, so a synthesis computes 4,097 of those rows and the other 248,191 never. The audio decoder's code embeddings
hold 4,096 rows for each of the 10 codebooks; a frame's codebook 0 is the main token itself, which the text model embeds
from its vocabulary, and the official code looks up the rows of codebooks 1 to 9 alone, never the first 4,096. The
model is to run whole on an Apple M5, whose Metal takes at most 25,559 MiB, and with its dense weights on an RTX 2080's
8 GB.

## Decision

The converter writes the head's rows of the end id and the semantic ids alone, in the order of their ids, which the file
records, and the audio decoder's embeddings of codebooks 1 to 9. A file without them computes what the checkpoint
computes.

The alternatives were turned down:

- Keeping the whole head. The rows no request reads would take memory where the model is tightest, 0.55 GB of Q8_0 on
  the GPU of a device that holds the model whole.
- Keeping codebook 0's embeddings. Nothing reads them.

## Consequences

A use of the head's other rows, such as the model writing text, needs a new layout and the files converted again.
