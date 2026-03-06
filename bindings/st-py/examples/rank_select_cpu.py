import spiraltorch as st


def main() -> None:
    x = st.Tensor(
        2,
        5,
        [
            1.0,
            3.0,
            2.0,
            3.0,
            -1.0,
            0.0,
            -2.0,
            5.0,
            4.0,
            5.0,
        ],
    )

    top = st.rank_select_cpu(x, kind="topk", k=2)
    print("topk values=", top["values"].tolist())
    print("topk indices=", top["indices"])

    bottom = st.rank_select_cpu(x, kind="bottomk", k=2)
    print("bottomk values=", bottom["values"].tolist())
    print("bottomk indices=", bottom["indices"])

    mid = st.rank_select_cpu(x, kind="midk", k=2)
    print("midk values=", mid["values"].tolist())
    print("midk indices=", mid["indices"])


if __name__ == "__main__":
    main()
