"""The independent reference must preserve leading axes and changed widths."""
import os
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import eager


class Reference(unittest.TestCase):
    def test_rectangular_linear_on_cpu_and_real_mps(self):
        self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "0")
        self.assertTrue(torch.backends.mps.is_available())
        plan = {"stages":[{"kind":"linear","weight":0,"bias":1,"gelu":False}]}
        with torch.inference_mode():
            for device in ("cpu","mps"):
                params = [torch.tensor(v,device=device,dtype=torch.float32) for v in
                          ([[1,0,0,2],[0,1,0,0],[0,0,1,0]],[.1,.2,.3,.4])]
                for shape in ((2,3),(1,2,3)):
                    x = torch.arange(6,device=device,dtype=torch.float32).reshape(shape)
                    out = eager(torch,plan,x,params)
                    self.assertEqual(tuple(out.shape),(*shape[:-1],4))
                    expected = torch.tensor([.1,1.2,2.3,.4,3.1,4.2,5.3,6.4])
                    torch.testing.assert_close(out.cpu().flatten(),expected,atol=1e-6,rtol=1e-6)
                out = eager(torch,plan,torch.tensor([0,1,2],device=device,dtype=torch.float32),params)
                self.assertEqual(tuple(out.shape),(4,))
                torch.testing.assert_close(out.cpu(),torch.tensor([.1,1.2,2.3,.4]),atol=1e-6,rtol=1e-6)


if __name__ == "__main__": unittest.main()
