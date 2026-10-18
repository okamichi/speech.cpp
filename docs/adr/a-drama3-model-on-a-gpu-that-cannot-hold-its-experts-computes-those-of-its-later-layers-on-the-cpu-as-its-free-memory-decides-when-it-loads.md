# A Drama 3 model on a GPU that cannot hold its experts computes those of its later layers on the CPU, as its free memory decides when it loads

## Context

Drama 3's Q4_K file holds 20.8 GB, of which its routed experts, 256 in each of 40 layers, hold 18.1 GB and the rest
of the model 2.7 GB. A row of the text model reads 8 experts of each layer, about 14 MB of a layer and 0.57 GB of a row,
and about 0.9 GB outside the experts. An Apple M5 lets Metal hold 25.5 GB and holds the whole file; an RTX 2080 holds 8
GB and refuses to load it. A request also needs memory beside the weights: the keys and values of the text model's 10
attention layers and the audio decoder's 13 layers, in float32, for up to 8,192 prompt rows and 2,048 frames, 0.9 GB at
their most, and the buffers of the graphs, 1.0 GB at their largest.

ggml's scheduler runs a graph over several backends and puts each node on the one that holds the weights it reads;
neither Metal nor Vulkan computes from the CPU's buffers, so a layer whose experts the CPU holds becomes a split on the
CPU between two on the GPU. On an RTX 2080 with Vulkan and an i9-9900K, the Q4_K file, a Japanese sentence of 46 prompt
rows, on 2026-10-15, a row of the text model took 9.8 ms on the GPU without any experts, and with the experts on the
CPU:

| Variant | Row | RTFx | Peak of the GPU's memory in use, the desktop's 0.6 GiB included |
|---|---|---|---|
| every layer's experts on the CPU, ggml's threads started for each graph | 95 ms | 0.4 | |
| the same with a pool of threads that waits between graphs, 4 or 8 threads | 55 to 69 ms | 0.5 to 0.6 | 3.6 GiB |
| the experts of the first 8 layers kept on the GPU | 49 to 50 ms | 0.7 | 6.9 GiB |
| the first 10 layers | 46 to 48 ms | 0.6 to 0.75 | 7.6 GiB |
| the scheduler's offload of the prompt's experts to the GPU | 67 ms, the prompt 2.5 s against 1.0 s | 0.4 | |
| the CPU's tensors of the graph in Vulkan's pinned host memory | 46.0 against 47.8 ms (10 layers) | | |

A layer whose experts the CPU holds costs about 1.15 ms of a row: 0.55 ms of reading its 8 experts and two round
trips of about 0.3 ms each, the wait for a submission to end rather than the copy, which pinned memory does not shorten.

## Decision

- **A model loaded on a GPU keeps there the experts of as many layers, from the first, as the GPU's free memory holds**
  when it loads, beside the rest of the model's weights, the caches of the largest request and a reserve for the graphs
  and the backend measured at the worst request. The model allocates those caches and the buffers of its largest graphs
  as it loads, so that a model loaded after it, as `speech serve` loads one of each task, cannot take their memory. The
  CPU's memory holds the experts of the later layers, which the CPU
  computes with the load's threads, in a pool that waits between graphs. A GPU that holds every expert computes them
  all; one that cannot hold the rest of the model beside the reserve refuses the load as out of memory, naming what it
  has free and what the model needs.
- **The placement is said, not hidden**: the load logs a warning naming the layers computed on the CPU, and the
  model's information reports the CPU's threads, as it does for a model on a GPU that computes part of a request on the
  CPU.
- **The request option `experts_on_cpu`, true by default**, lets a request refuse a model that holds some of its
  experts on the CPU, as a value out of the range such a model takes, naming the option, for a caller that wants the GPU's speed or nothing; the same request then runs with true. A model
  whose GPU holds every expert, or one loaded on the CPU, takes either value.
- **The prompt's experts are computed on the CPU too**, without the scheduler's offload.

The alternatives were turned down:

- Every layer's experts on the CPU wherever the model does not fit. A layer whose experts the GPU keeps costs nothing
  of the 1.15 ms, so the GPU's free memory is worth filling: 46 ms a row with 10 layers kept against 56 ms with none.
- Whole layers on the CPU, attention and Gated DeltaNet included, as llama.cpp's layer split runs a model. It crosses
  between the devices twice a row instead of twice a layer, but the CPU then reads each such layer's 25 MB outside the
  experts as well as its experts' 14 MB, on a CPU whose memory carries about 30 GB/s against the GPU's 448 GB/s.
- Choosing the layers by trying to allocate them and keeping what fits. The weights would take the memory that a
  request's caches and graphs need later: with the experts of 12 layers kept, a request of 1,070 rows ran out of memory
  for its caches.
- The scheduler's offload of the prompt's experts to the GPU, which copies each layer's experts over for every block
  of the prompt: a prompt of 46 rows took 2.5 s against 1.0 s on the CPU.
- A load parameter instead of a request option. The load parameters are the same for every family, and a request
  option reaches the worker's and the server's flags as every option does.

## Consequences

On the RTX 2080 a model holds the experts of 5 layers on the GPU and speaks at 0.5 to 0.6 times real time, with its
first audio after 1.1 to 1.3 s, where a GPU of 12 or 16 GB keeps about 18 or 27 layers' experts and comes nearer real
time. The model holds 6.8 GiB of the GPU's memory from its load on, beside the desktop's 0.6 GiB, and the worst
request, a voice of about 8,000 rows of recordings, took it to 6.9 GiB. The CPU computes its
experts in its own arithmetic, which differs from the GPU's: the text model's check against the official dumps gives
the experts' output at 67.9 dB with them on the CPU against 65.4 dB on Metal, in F32. A prompt of 8,000 rows of
references takes about a minute before the first audio with most experts on the CPU. Every model holds the memory of
its largest request from its load on, about 2.1 GB beyond its weights on the M5, whatever requests it serves.
