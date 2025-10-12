use ndarray::{Array2, ArrayView2};

pub fn topk2d_cpu(x: ArrayView2<'_, f32>, k: usize) -> (Array2<f32>, Array2<i32>) {
    let (rows, cols) = x.dim();
    let mut vals = Array2::<f32>::zeros((rows, k));
    let mut idxs = Array2::<i32>::zeros((rows, k));
    for r in 0..rows {
        let mut v: Vec<(f32, usize)> = (0..cols).map(|c| (x[(r,c)], c)).collect();
        v.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        for t in 0..k {
            vals[(r,t)] = v[t].0;
            idxs[(r,t)] = v[t].1 as i32;
        }
    }
    (vals, idxs)
}

pub fn bottomk2d_cpu(x: ArrayView2<'_, f32>, k: usize) -> (Array2<f32>, Array2<i32>) {
    let (rows, cols) = x.dim();
    let mut vals = Array2::<f32>::zeros((rows, k));
    let mut idxs = Array2::<i32>::zeros((rows, k));
    for r in 0..rows {
        let mut v: Vec<(f32, usize)> = (0..cols).map(|c| (x[(r,c)], c)).collect();
        v.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
        for t in 0..k {
            vals[(r,t)] = v[t].0;
            idxs[(r,t)] = v[t].1 as i32;
        }
    }
    (vals, idxs)
}

pub fn midk2d_cpu(x: ArrayView2<'_, f32>, k: usize) -> (Array2<f32>, Array2<i32>) {
    let (rows, cols) = x.dim();
    assert!(k<=cols);
    let drop_each = (cols - k)/2;
    let (top_vals, top_idx) = topk2d_cpu(x.view(), drop_each);
    let (bot_vals, bot_idx) = bottomk2d_cpu(x.view(), drop_each);
    // mark extremes
    let mut mask = vec![vec![false; cols]; rows];
    for r in 0..rows {
        for t in 0..drop_each {
            mask[r][top_idx[(r,t)] as usize] = true;
            mask[r][bot_idx[(r,t)] as usize] = true;
        }
    }
    let mut vals = Array2::<f32>::zeros((rows, k));
    let mut idxs = Array2::<i32>::zeros((rows, k));
    for r in 0..rows {
        let mut out=0usize;
        for c in 0..cols {
            if !mask[r][c] {
                if out<k { vals[(r,out)]=x[(r,c)]; idxs[(r,out)]=c as i32; out+=1; }
            }
        }
    }
    (vals, idxs)
}
