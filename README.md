# Deconvolve
Estimate IR from two recordings uding PyTorh 1D convolution.

Given two recordings of the same audio source: `A` and `B`, this code estimated a convolution kernel `g` such as that `B = g * A`.

For example, if you have a recording made by two different microphones (e.g. piezo pick-up and a dynamic/condenser mic), you can estimate a filter to be applied to the piezo pickup to make is sound as a dynamic/condenser mic.

## How to use

```shell
python torch_deconvolve.py source.wav target.wav ir.wav
```

This will run the optimization process to find the convolution kernel and then save it as `ir.wav`.

Both recording must be mono, the same sample rate and preferably the same length. The longer the recording is, the longer it will take to run the optimization, but the result may be more accurate.

> For better result try to use harmonically rich audio sources that cover a wide frequency range.

For more parameters run:
```shell
python torch_deconvolve.py --help
```
## Note
An optimization may take a while. It helps if you have a CUDA-enabled GPU.
