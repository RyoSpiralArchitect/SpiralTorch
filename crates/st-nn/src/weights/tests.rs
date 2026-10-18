use super::*;
use safetensors::tensor::{serialize, TensorView};
use std::io::{Cursor, Read, Write};

fn fixture(dtype: Dtype, shape: &[usize], data: &[u8]) -> Vec<u8> {
    serialize(
        [(
            "weight",
            TensorView::new(dtype, shape.to_vec(), data).unwrap(),
        )],
        Some([("format".into(), "pt".into())].into()),
    )
    .unwrap()
}

fn raw(header: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn parse(bytes: &[u8]) -> Result<SafeTensorArchive, WeightLoadError> {
    SafeTensorArchive::from_bytes(bytes, WeightLoadLimits::default())
}

#[test]
fn official_writer_f32_f16_bf16_and_native_matmul() {
    let expected = [1.0f32, -2.0, 0.5, 4.0];
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        let data: Vec<u8> = expected
            .iter()
            .flat_map(|&v| match dtype {
                Dtype::F32 => v.to_le_bytes().to_vec(),
                Dtype::F16 => f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                Dtype::BF16 => bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect();
        let archive = parse(&fixture(dtype, &[2, 2], &data)).unwrap();
        assert_eq!(archive.names(), ["weight"]);
        assert_eq!(archive.metadata()["format"], "pt");
        assert_eq!(archive.info("weight").unwrap().dtype, dtype.to_string());
        let tensor = archive.tensor("weight").unwrap();
        assert_eq!(tensor.read_values().unwrap(), expected);
        let x = st_tensor::Tensor::from_vec(1, 2, vec![2., 3.]).unwrap();
        assert_eq!(
            x.matmul(&tensor.to_tensor().unwrap()).unwrap().data(),
            [3.5, 8.]
        );
    }
}

#[test]
fn shapes_scalar_empty_and_explicit_rank_conversion() {
    for shape in [vec![], vec![1], vec![1, 1, 1]] {
        let archive = parse(&fixture(Dtype::F32, &shape, &2f32.to_le_bytes())).unwrap();
        let tensor = archive.tensor("weight").unwrap();
        assert_eq!(tensor.shape(), shape);
        assert!(tensor.to_tensor().is_err());
        assert_eq!(
            tensor.reshape(&[1, 1]).unwrap().to_tensor().unwrap().data(),
            [2.]
        );
    }
    let archive = parse(&fixture(Dtype::F16, &[2, 0, 3], &[])).unwrap();
    assert_eq!(archive.info("weight").unwrap().elements, 0);
    assert_eq!(archive.tensor("weight").unwrap().shape(), [2, 0, 3]);
    assert!(archive
        .tensor("weight")
        .unwrap()
        .read_values()
        .unwrap()
        .is_empty());
    assert!(parse(&raw("{}", &[])).unwrap().names().is_empty());
}

#[test]
fn duplicate_names_fields_and_metadata_are_rejected() {
    let t = r#"{"dtype":"F32","shape":[0],"data_offsets":[0,0]}"#;
    for header in [
        format!(r#"{{"w":{t},"w":{t}}}"#),
        format!(r#"{{"w":{t},"\u0077":{t}}}"#),
        r#"{"w":{"dtype":"F32","dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#.into(),
        r#"{"w":{"dtype":"F32","shape":[0],"shape":[1],"data_offsets":[0,0]}}"#.into(),
        r#"{"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0],"unknown":0}}"#.into(),
        r#"{"__metadata__":{"format":"pt","format":"tf"}}"#.into(),
        r#"{"__metadata__":{"format":1}}"#.into(),
        r#"{"__metadata__":null}"#.into(),
    ] {
        assert!(parse(&raw(&header, &[])).is_err(), "{header}");
    }
}

#[test]
fn malformed_header_and_overflows_never_panic() {
    for bytes in [
        vec![],
        vec![0; 7],
        u64::MAX.to_le_bytes().to_vec(),
        raw(" {} ", &[]),
        raw("{}[]", &[]),
        raw("[]", &[]),
        raw(
            r#"{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}"#,
            &[],
        ),
        raw(
            &format!(
                r#"{{"w":{{"dtype":"F32","shape":[{},2],"data_offsets":[0,0]}}}}"#,
                usize::MAX
            ),
            &[],
        ),
        raw(
            &format!(
                r#"{{"w":{{"dtype":"F32","shape":[1],"data_offsets":[0,{}]}}}}"#,
                usize::MAX
            ),
            &[],
        ),
    ] {
        assert!(parse(&bytes).is_err());
    }
}

#[test]
fn official_parser_rejects_holes_overlaps_trailing_data_and_size_mismatch() {
    for header in [
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    ] {
        assert!(parse(&raw(header, &[0; 8])).is_err(), "{header}");
    }
}

#[test]
fn nonfinite_all_supported_dtypes_fail_only_when_materialized() {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let data = match dtype {
                Dtype::F32 => value.to_le_bytes().to_vec(),
                Dtype::F16 => f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                Dtype::BF16 => bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            };
            let archive = parse(&fixture(dtype, &[], &data)).unwrap();
            assert_eq!(archive.info("weight").unwrap().elements, 1);
            assert!(matches!(
                archive.tensor("weight"),
                Err(WeightLoadError::NonFinite { index: 0, .. })
            ));
        }
    }
}

#[test]
fn unsupported_dtype_stays_inspectable_but_never_silently_casts() {
    let archive = parse(&fixture(Dtype::I64, &[1], &u64::MAX.to_le_bytes())).unwrap();
    assert_eq!(archive.info("weight").unwrap().dtype, "I64");
    assert!(matches!(
        archive.tensor("weight"),
        Err(WeightLoadError::Dtype(_))
    ));
    assert!(matches!(
        archive.info("missing"),
        Err(WeightLoadError::Missing(_))
    ));
    assert!(matches!(
        archive.tensor("missing"),
        Err(WeightLoadError::Missing(_))
    ));
}

#[test]
fn admission_limits_and_exact_boundaries() {
    let bytes = fixture(Dtype::F32, &[1, 1], &3f32.to_le_bytes());
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let limits = WeightLoadLimits {
        max_file_bytes: bytes.len(),
        max_header_bytes: header_len,
        max_tensors: 1,
        max_rank: 2,
        max_tensor_elements: 1,
    };
    assert!(SafeTensorArchive::from_bytes(&bytes, limits)
        .unwrap()
        .tensor("weight")
        .is_ok());
    for changed in [
        WeightLoadLimits {
            max_file_bytes: bytes.len() - 1,
            ..limits
        },
        WeightLoadLimits {
            max_header_bytes: header_len - 1,
            ..limits
        },
        WeightLoadLimits {
            max_rank: 1,
            ..limits
        },
    ] {
        assert!(SafeTensorArchive::from_bytes(&bytes, changed).is_err());
    }
    let no_values = WeightLoadLimits {
        max_tensor_elements: 0,
        ..limits
    };
    assert!(matches!(
        SafeTensorArchive::from_bytes(&bytes, no_values)
            .unwrap()
            .tensor("weight"),
        Err(WeightLoadError::Limit(_))
    ));
    let t = r#"{"dtype":"F32","shape":[0],"data_offsets":[0,0]}"#;
    let bytes = raw(&format!(r#"{{"a":{t},"b":{t}}}"#), &[]);
    assert!(SafeTensorArchive::from_bytes(
        &bytes,
        WeightLoadLimits {
            max_tensors: 1,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn invalid_limits_rejected_even_for_empty_archive() {
    for limits in [
        WeightLoadLimits {
            max_file_bytes: usize::MAX,
            ..Default::default()
        },
        WeightLoadLimits {
            max_file_bytes: 7,
            ..Default::default()
        },
        WeightLoadLimits {
            max_header_bytes: 100_000_001,
            ..Default::default()
        },
        WeightLoadLimits {
            max_header_bytes: 0,
            ..Default::default()
        },
        WeightLoadLimits {
            max_rank: 129,
            ..Default::default()
        },
        WeightLoadLimits {
            max_tensors: 0,
            ..Default::default()
        },
        WeightLoadLimits {
            max_tensor_elements: usize::MAX,
            ..Default::default()
        },
    ] {
        assert!(SafeTensorArchive::from_bytes(&raw("{}", &[]), limits).is_err());
    }
}

#[test]
fn snapshot_is_owned_and_returned_tensors_are_independent() {
    let mut bytes = fixture(Dtype::F32, &[1, 1], &2f32.to_le_bytes());
    let archive = parse(&bytes).unwrap();
    bytes.fill(0);
    let tensor = archive.tensor("weight").unwrap();
    let mut legacy = tensor.to_tensor().unwrap();
    legacy.data_mut()[0] = 8.;
    assert_eq!(tensor.read_values().unwrap(), [2.]);
    assert_eq!(
        archive.tensor("weight").unwrap().read_values().unwrap(),
        [2.]
    );
}

#[test]
fn file_snapshot_survives_replacement_and_matches_bytes() {
    let bytes = fixture(Dtype::F32, &[1, 1], &2f32.to_le_bytes());
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    let archive = SafeTensorArchive::open(file.path(), WeightLoadLimits::default()).unwrap();
    std::fs::write(file.path(), []).unwrap();
    assert_eq!(
        archive.tensor("weight").unwrap().read_values().unwrap(),
        [2.]
    );
    assert!(SafeTensorArchive::open(file.path(), WeightLoadLimits::default()).is_err());
    assert!(
        SafeTensorArchive::open(file.path().join("missing"), WeightLoadLimits::default()).is_err()
    );
}

#[cfg(unix)]
#[test]
fn fifo_without_writer_is_rejected_without_blocking() {
    const CHILD: &str = "ST_WEIGHT_FIFO_TEST_PATH";
    if let Some(path) = std::env::var_os(CHILD) {
        assert!(matches!(
            SafeTensorArchive::open(path, WeightLoadLimits::default()),
            Err(WeightLoadError::Format(_))
        ));
        return;
    }
    use std::os::unix::ffi::OsStrExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("no-writer");
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: cpath owns a valid NUL-terminated path for the duration of the call.
    assert_eq!(unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) }, 0);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "weights::tests::fifo_without_writer_is_rejected_without_blocking",
        ])
        .env(CHILD, &path)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        if start.elapsed() > Duration::from_secs(5) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("opening a FIFO without a writer blocked");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn file_growth_and_truncation_fail_closed() {
    let bytes = fixture(Dtype::F32, &[1, 1], &2f32.to_le_bytes());
    for declared_len in [bytes.len() - 1, bytes.len() + 1] {
        assert!(SafeTensorArchive::read_file(
            &mut Cursor::new(&bytes),
            declared_len,
            WeightLoadLimits::default()
        )
        .is_err());
    }
}

#[test]
fn invalid_header_rejected_before_file_payload_read() {
    struct HeaderOnly(Cursor<Vec<u8>>);
    impl Read for HeaderOnly {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            assert!(
                (self.0.position() as usize) < self.0.get_ref().len(),
                "must not read payload"
            );
            self.0.read(buf)
        }
    }
    let mut reader = HeaderOnly(Cursor::new(raw(
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,999]}}"#,
        &[],
    )));
    let declared_len = reader.0.get_ref().len() + 4;
    assert!(
        SafeTensorArchive::read_file(&mut reader, declared_len, WeightLoadLimits::default())
            .is_err()
    );
}
