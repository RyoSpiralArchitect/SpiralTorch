import spiraltorch as st


def main() -> None:
    for device in ("mps", "wgpu", "cuda", "hip", "cpu"):
        out = st.plan("topk", 128, 8192, 64, device=device)
        print(f"device={device} choice={out['choice']}")


if __name__ == "__main__":
    main()

