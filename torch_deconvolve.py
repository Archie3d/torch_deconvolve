import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
import numpy as np

def load_wav(file_path):
    """
    Load a WAV file from a given path.
    """
    sample_rate, signal = wavfile.read(file_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # Convert to mono if stereo

    dtype = signal.dtype
    signal = signal.astype(np.float32)

    if dtype == np.int32:
        signal /= 2147483648
    elif dtype == np.int16:
        signal /= 32768
    elif signal.dtype == np.uint8:
        signal = (signal / 255) - 0.5

    return sample_rate, signal


def save_wav(file_path, sample_rate, signal):
    """
    Save WAV audio as PCM24 bit.
    We use wavio to write 3-bytes packer 24-bits data instead of 4-bytes
    """
    import wavio
    wavio.write(file_path, signal, sample_rate, sampwidth=3)


def make_spectral_loss(fft_length, hop_length, device):
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=fft_length,
        hop_length=hop_length,
        power=1,
        center=False,
        normalized=True,
    ).to(device)

    def loss_fn(x, y):
        loss = 0.0

        s_x = spectrogram(x)
        s_y = spectrogram(y)

        # If we don't use log loss, thr resulting IR will have an HF noise in it
        log_s_x = torch.log(s_x + 1e-12)
        log_s_y = torch.log(s_y + 1e-12)
        loss += torch.mean(torch.abs(log_s_x - log_s_y))

        return loss

    return loss_fn


def nextPowerOf2(n):
    """
    Returnes the smallest power of 2 that is bigger than n
    """
    assert n > 0
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def optimize(
        source,
        target,
        ir_length=2048,
        num_epochs=100000,
        patience=10,
        learning_rate=1e-3,
        improvement_threshold=1e-4,
        device='cpu'
    ):
    """
    Perform optimization to find an IR convolution kernel given the source and the target signals.
    """

    assert num_epochs > 100
    assert ir_length >= 16

    # Ensure signals are of the same length
    N = min(len(source), len(target))

    source_tensor = torch.tensor(source[:N], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    target_tensor = torch.tensor(target[:N], dtype=torch.float32, device=device)

    # Pad the source audio so that it has the same length after the convolution
    source_tensor = torch.nn.functional.pad(source_tensor, (0, ir_length - 1))

    # Define 1D convolution kernel, initialized with random weights
    kernel = nn.Parameter(torch.randn(1, 1, ir_length, requires_grad=True).to(device))

    # Exponential decay to force IR transient at its start
    decay = torch.exp(-6 * torch.arange(0, ir_length) / ir_length)
    decay = decay.to(device)

    w = nextPowerOf2(ir_length)
    loss_fn = make_spectral_loss(w, w // 4, device)

    optimizer = optim.Adam([kernel], lr=learning_rate)


    prev_loss = None
    best_loss = None
    best_kernel = None
    best_epoch = None
    current_patience = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        output = nn.functional.conv1d(source_tensor, kernel * decay)

        loss = loss_fn(output.squeeze(), target_tensor)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if best_loss == None or best_loss > loss:
            best_epoch = epoch
            best_loss = loss
            best_kernel = kernel.detach()

        # Progress logging
        if (epoch + 1) % 100 == 0:
            if prev_loss == None:
                delta = 1.0
            else:
                delta = (prev_loss - loss.item()) / loss.item()

            progress = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}, Improvement: {delta * 100:.2f}% "
            progress += '[' + ('.' * current_patience) + (' ' * (patience - current_patience)) + ']' + (' ' * 5)
            print(progress, end='\r')

            if delta < improvement_threshold:
                current_patience += 1
                if current_patience > patience:
                    break
            else:
                prev_loss = loss.item()
                current_patience = 0

    # Extract the learned kernel
    result = best_kernel * decay
    result = result.cpu().numpy().squeeze()

    return result, best_loss, best_epoch

#===============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='torch_deconvolve',
        description='Estimare IR filter given a source and a target signals'
    )
    parser.add_argument('source', help='Source wav file')
    parser.add_argument('target', help='Target wav file')
    parser.add_argument('ir', help='Output wav file to save estimated IR')
    parser.add_argument('-l', '--length', default=2048, help='IR length in number of samples, default is 2048')
    parser.add_argument('-t', '--threshold', default=0.01, help='Improvement threshold in %%, default is 0.01')
    parser.add_argument('-e', '--epochs', default=100000, help='Maximum number of epochs to run, default is 100000')

    args = parser.parse_args()

    print(f"Loading source audio from {args.source}...")
    source_sample_rate, source = load_wav(args.source)
    print(f"Loading target audio from {args.target}...")
    target_sample_rate, target = load_wav(args.target)

    if source_sample_rate != target_sample_rate:
        print(f"Source audio sample rate ({source_sample_rate}) does not match the target sample rate ({target_sample_rate})")
        return -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running optimization on {device} for IR of {args.length} samples...")

    ir, best_loss, best_epoch = optimize(
        source,
        target,
        ir_length=args.length,
        improvement_threshold=float(args.threshold) * 0.01,
        num_epochs=args.epochs,
        device=device
    )

    print(f"\n\nBest loss: {best_loss:.6f} on epoch {best_epoch}")

    print(f"Saving IR as '{args.ir}'...")
    save_wav(args.ir, source_sample_rate, ir)


if __name__ == "__main__":
    main()
