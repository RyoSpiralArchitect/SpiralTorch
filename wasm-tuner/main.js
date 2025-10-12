// Skeleton: replace with real WebGPU tuning. Exports minimal tuner.json for the pipeline test.
document.getElementById('run').onclick = () => {
  const json = {
    topk: [
      { rows: 1024, cols: 65536, k: 1024, subgroup: true, use_2ce: true, wg:256, kl:32, ch:8192, time_ms: 0.8 }
    ],
    midk: [
      { rows: 1024, cols: 65536, k: 32768, subgroup: true, scan_wg:256, tile_cols:4096, two_ce:true, time_ms: 1.2 }
    ]
  };
  const blob = new Blob([JSON.stringify(json,null,2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tuner.json';
  a.click();
  document.getElementById('log').textContent = 'Exported tuner.json (stub). Replace with real measurements.';
};
