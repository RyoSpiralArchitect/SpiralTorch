import spiraltorch as st


class EntrypointDemoPlugin:
    def metadata(self) -> dict:
        return {
            "id": "demo_entrypoint_plugin",
            "version": "0.0.1",
            "capabilities": ["Telemetry"],
        }

    def on_load(self) -> None:
        st.plugin.register_service("demo_entrypoint_plugin.hello", "world")

    def on_event(self, event: dict) -> None:
        _ = event


plugin = EntrypointDemoPlugin()
