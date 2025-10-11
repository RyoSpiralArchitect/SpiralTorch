use pyo3::prelude::*;

#[pymodule]
fn spiraltorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "1.3.21")?;
    Ok(())
}
