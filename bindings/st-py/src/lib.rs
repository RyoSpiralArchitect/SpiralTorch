use pyo3::prelude::*;

#[pymodule]
fn spiraltorch_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "1.3.7")?;
    m.add_function(wrap_pyfunction!(backends, m)?)?;
    Ok(())
}

#[pyfunction]
fn backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    #[cfg(all(feature="mps", target_os="macos"))] { v.push("mps"); }
    #[cfg(feature="wgpu")] { v.push("wgpu"); }
    #[cfg(feature="cuda")] { v.push("cuda"); }
    v
}
