# The fork keeps its own option numbers and C API major version, and appends the options upstream adds after its own

## Context

This repository is a fork of nyosegawa/speech.cpp that merges upstream's main from time to time. The fork's option
`irodori_compatible` holds the value 44 in `speech_option`, which upstream gives to `decode_on_cpu`, so every option
from 44 on has a value one higher in the fork than upstream, and the fork's C API is at major version 4 where
upstream's is at 3. Upstream adds each new option at the end of its enum, at a value the fork already uses, so every
merge meets the same collision in the enum and in the library's vocabulary, whose order is the options' values.

## Decision

A merge keeps the fork's values as they are and appends the options upstream added since the last merge after the
fork's last option, in upstream's order. The fork's C API stays at major version 4. A merge that brings options or
functions raises the fork's minor version, and every declaration upstream added gives the fork's version in its
"Added in", not upstream's. A program built against upstream's header does not run against the fork's library;
programs that reach the models through the worker or the HTTP server name options by their text and do not notice.

The alternatives were turned down:

- Taking upstream's values and moving `irodori_compatible` to the end changes the values that programs built against
  the fork's 4.x header use, which raises the major version again, and the next option upstream adds takes the value
  the fork's option then holds, so the values would match upstream's only until the next merge.

## Consequences

Each merge of upstream resolves by hand the enum, the vocabulary and the C API check's list of option names, all in
the order above.
