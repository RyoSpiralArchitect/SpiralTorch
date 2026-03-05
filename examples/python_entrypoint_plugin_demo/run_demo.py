import spiraltorch as st


def main() -> None:
    loaded = st.plugin.load_entrypoints(replace=True)
    print("loaded:", loaded)
    for plugin_id in loaded:
        meta = st.plugin.plugin_metadata(plugin_id)
        print(f"meta[{plugin_id}]:", meta)
    print("services:", st.plugin.list_services())
    print("hello:", st.plugin.get_service("demo_entrypoint_plugin.hello"))


if __name__ == "__main__":
    main()

