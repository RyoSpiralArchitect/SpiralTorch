fn main() {
    // Cross-platform: PyO3 が各プラットフォームに必要なリンカ設定を追加
    pyo3_build_config::add_extension_module_link_args();

    // macOS では明示的に cdylib のリンク引数にも差し込む（cargo のバージョンや構成差異対策）
    #[cfg(target_os = "macos")]
    {
        // 2通りの指定を両方入れておく（どちらかが環境差で効くケースがある）
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
        // デバッグ用: 本当に build.rs が走っているか確認したいときにログを見る
        println!("cargo:warning=injecting -undefined dynamic_lookup for macOS");
    }
}
