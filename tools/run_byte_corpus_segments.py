"""Compatibility CLI for the installed SpiralTorch segment execution primitive."""


def run(request, output, stops, **kwargs):
    from spiraltorch import run_byte_corpus_segments
    return run_byte_corpus_segments(request, output, stops, **kwargs)


def main():
    from spiraltorch.byte_corpus_runs import main as packaged_main
    packaged_main()


if __name__ == "__main__":
    main()
