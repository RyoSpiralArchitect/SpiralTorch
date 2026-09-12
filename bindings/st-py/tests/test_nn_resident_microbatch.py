"""Cross-input GPU accumulation through the public API, independent scalar oracle."""
import math
import os
import unittest
import spiraltorch as st

def read(t): return t.snapshot().read_values()

class Surface(unittest.TestCase):
    def test_accumulator_is_learner_owned_not_publicly_constructible(self):
        from spiraltorch.nn import GraphGradientAccumulator
        with self.assertRaises(TypeError): GraphGradientAccumulator()

@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def near(self,a,b):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b): self.assertTrue(math.isfinite(x) and abs(x-y)<=2e-5+2e-4*abs(y),(x,y))

    def test_weighted_microbatches_reuse_and_recover(self):
        data = [([1.,-.5,1.5,-1.],[0.,1.,0.,1.]), ([.5,1.,-1.,2.],[0.,0.,1.,-100.]), ([2.,0.,0.,0.],[0.,-100.,-100.,-100.])]
        for policy,scale in [("exact",1.),("module_compatible",.25)]:
            for reduction in ["mean","sum"]:
                model = st.nn.Sequential(); model.add(st.nn.Scaler.from_gain("gain",st.Tensor(1,2,[0.,0.])))
                baseline = model.inference_plan([2,2,2]); learner = baseline.compile_graph_learner_wgpu(gradient_policy=policy)
                device = learner.tensor_device(); self.assertNotEqual(device.adapter_info()["device_type"],"Cpu")
                batches = [(device.upload([2,2,2],[v for x in xs for v in [x,x]]),device.upload([2,2],ys)) for xs,ys in data]
                objective = st.nn.CrossEntropyWithLogits(reduction=reduction,label_smoothing=.1)
                accumulator = learner.gradient_accumulator(); held=[]; gain=[0.,0.]
                with self.assertRaises(ValueError): accumulator.parameter_gradient_tensors()
                with self.assertRaises(ValueError): learner.sgd_accumulated(accumulator,0.)
                for step in range(32):
                    learner.zero_accumulator(accumulator); self.assertEqual(accumulator.parameter_generation,step)
                    expected=[0.,0.]
                    for (xs,ys),(x,y) in zip(data,batches):
                        learner.set_input_tensor(x); forward=learner.forward()
                        pair=objective.evaluate_resident(forward.prediction_tensor(),y)
                        gradient=learner.backward(forward,pair.prediction_gradient_tensor())
                        count=sum(y!=-100. for y in ys)
                        learner.accumulate(accumulator,gradient,count/8 if reduction=="mean" else 1/8)
                        for value,label in zip(xs,ys):
                            if label==-100.: continue
                            p=1/(1+math.exp(-value*(gain[0]-gain[1]))); target=.95 if label==0. else .05
                            expected[0]+=value*(p-target)/8; expected[1]+=value*((1-p)-(1-target))/8
                    self.assertEqual(len(accumulator),3)
                    held.append((accumulator.parameter_gradient_tensors()[0],expected.copy()))
                    learner.sgd_accumulated(accumulator,.1); held[-1]+=(learner.update_snapshot(),)
                    for i in range(2): gain[i]-=.1*scale*expected[i]
                before=learner.parameter_snapshot().read_plan()
                self.assertEqual(baseline.apply_parameters_to(model,before),1)
                self.near(read(model(batches[0][0])),[value*g for value in data[0][0] for g in gain])
                with self.assertRaises(ValueError): learner.sgd_accumulated(accumulator,0.)
                learner.zero_accumulator(accumulator)
                with self.assertRaises(ValueError): learner.accumulate(accumulator,gradient,1.)
                foreign=baseline.compile_graph_learner_wgpu(gradient_policy=policy)
                with self.assertRaises(ValueError): foreign.zero_accumulator(accumulator)
                learner.set_input_tensor(batches[0][0]); forward=learner.forward()
                bad=objective.evaluate_resident(forward.prediction_tensor(),device.upload([2,2],[0.,.5,0.,1.]))
                broken=learner.backward(forward,bad.prediction_gradient_tensor())
                learner.accumulate(accumulator,broken,0.)
                good_pair=objective.evaluate_resident(forward.prediction_tensor(),batches[0][1])
                good=learner.backward(forward,good_pair.prediction_gradient_tensor())
                with self.assertRaises(ValueError): learner.accumulate(accumulator,good,float("nan"))
                self.assertEqual(len(accumulator),1)
                learner.accumulate(accumulator,good,1.)
                rejected_tensor=accumulator.parameter_gradient_tensors()[0]
                learner.sgd_accumulated(accumulator,0.)
                with self.assertRaises(ValueError): learner.update_snapshot().read()
                self.assertEqual(learner.parameter_snapshot().read_plan().to_json(),before.to_json())
                learner.zero_accumulator(accumulator); forward=learner.forward()
                pair=objective.evaluate_resident(forward.prediction_tensor(),batches[0][1])
                good=learner.backward(forward,pair.prediction_gradient_tensor())
                learner.accumulate(accumulator,good,1.); learner.sgd_accumulated(accumulator,0.)
                self.assertEqual(learner.update_snapshot().read(),34)
                self.assertEqual(learner.parameter_snapshot().read_plan().to_json(),before.to_json())
                del learner,accumulator,model,device,foreign
                for i,(snapshot,expected,receipt) in enumerate(held,1):
                    self.near(read(snapshot),expected); self.assertEqual(receipt.read(),i)
                with self.assertRaises(ValueError): read(rejected_tensor)

if __name__ == "__main__": unittest.main()
