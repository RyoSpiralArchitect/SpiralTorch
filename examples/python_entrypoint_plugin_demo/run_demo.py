import spiraltorch as st


def main() -> None:
    loaded = st.plugin.reload_entrypoints(strict=False)
    print("loaded:", loaded)
    for plugin_id in loaded:
        meta = st.plugin.plugin_metadata(plugin_id)
        print(f"meta[{plugin_id}]:", meta)
    print("plugins:", st.plugin.list_plugins())
    print("services:", st.plugin.list_services())
    print("hello:", st.plugin.get_service("demo_entrypoint_plugin.hello"))
    unloaded = st.plugin.unload_entrypoints()
    print("unloaded:", unloaded)
    print("plugins after unload:", st.plugin.list_plugins())


if __name__ == "__main__":
    main()
