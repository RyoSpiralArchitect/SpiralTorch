import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, Sequential

session = st.SpiralSession(
    device="wgpu",
    curvature=-1.0,
    hyper_learning_rate=0.05,
    fallback_learning_rate=0.01,
)

densities = [st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 2, [1.0, 0.0])]
bary = session.barycenter(densities, weights=[0.5, 0.5])
hyper = session.hypergrad(*bary.density.shape())
session.align_hypergrad(hyper, bary)

model = Sequential([Linear(2, 2, name="layer")])
session.prepare_module(model)

trainer = session.trainer()
schedule = session.roundtable(
    rows=1,
    cols=2,
    psychoid=True,
    psychoid_log=True,
    psi=True,
    psi_log=True,
    collapse=True,
)
loss = MeanSquaredError()

dataset = st.dataset.from_vec(
    [
        (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 2, [0.0, 1.0])),
        (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 2, [1.0, 0.0])),
    ]
).shuffle(0xC0FFEE).batched(2).prefetch(2)

stats = session.train_epoch(trainer, model, loss, dataset, schedule)
print(f"roundtable avg loss {stats.average_loss:.6f} over {stats.batches} batches")
print(st.get_psychoid_stats())
