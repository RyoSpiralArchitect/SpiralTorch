import test from "node:test";
import assert from "node:assert/strict";
import {completedReads, routeOrder, routes, protocol, clockProbe} from "../bindings/st-wasm/tests/module_resident_intervals.mjs";

test("each forward waits for its own read and releases before the next", async () => {
  let active = null, calls = 0, clockCalls = 0;
  const events = [];
  const result = await completedReads({forwards: 3,
    now: () => (++clockCalls === 1 ? 10 : 16),
    forward() {
      assert.equal(active, null);
      active = ++calls; events.push("forward" + active); return active;
    },
    async read(value) {
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(active, value); events.push("read" + value);
      return new Float32Array([value]);
    },
    release(value) {assert.equal(active, value); events.push("free" + value); active = null;},
  });
  assert.deepEqual(events, ["forward1", "read1", "free1", "forward2", "read2", "free2", "forward3", "read3", "free3"]);
  assert.equal(clockCalls, 2);
  assert.equal(result.elapsed_ms, 6);
  assert.equal(result.completed_reads, 3);
  assert.deepEqual(result.outputs.map(value => Array.from(value)), [[1], [2], [3]]);
});

test("a failed read is released and never retried or followed by a new forward", async () => {
  let forwards = 0, released = 0;
  await assert.rejects(completedReads({forwards: 3,
    forward: () => ++forwards,
    read: async () => {throw Error("device lost");},
    release: () => released++,
  }), /device lost/);
  assert.equal(forwards, 1); assert.equal(released, 1);
});

test("default interval really reads every one of 256 outputs", async () => {
  let forwards = 0, reads = 0, released = 0, clock = 0;
  const result = await completedReads({now: () => ++clock,
    forward: () => ++forwards, read: async () => ++reads, release: () => released++,
  });
  assert.deepEqual([forwards, reads, released, result.outputs.length], [256, 256, 256, 256]);
  assert.equal(result.forwards, protocol.forwards);
});

test("invalid counts and non-positive or nonfinite clocks fail", async () => {
  const callbacks = {forward: () => 1, read: async () => 1, release: () => {}};
  for (const forwards of [0, -1, 0.5, true, NaN, Infinity]) {
    await assert.rejects(completedReads({...callbacks, forwards}), /forward count/);
  }
  for (const now of [() => 0, () => NaN, () => Infinity]) {
    await assert.rejects(completedReads({...callbacks, forwards: 1, now}), /interval clock/);
  }
});

test("fixed rotation covers every route and balances each retained position", () => {
  for (const seed of [17, 29, 43]) {
    const counts = Object.fromEntries(routes.map(route => [route, [0, 0, 0, 0]]));
    for (let matrix = 0; matrix < protocol.matrices; matrix++) {
      for (let block = protocol.warmup; block < protocol.warmup + protocol.samples; block++) {
        const order = routeOrder(matrix, seed, block);
        assert.deepEqual([...order].sort(), [...routes].sort());
        order.forEach((route, position) => counts[route][position]++);
      }
    }
    for (const row of Object.values(counts)) assert.deepEqual(row, [9, 9, 9, 9]);
  }
});

test("clock probe reports coarse edges, not zero duration", () => {
  let reads = 0;
  const clock = clockProbe(() => Math.floor(reads++ / 10) / 10);
  assert.equal(clock.positive_deltas_ms.length, 256);
  assert.ok(Math.abs(clock.minimum_positive_delta_ms - 0.1) < 1e-12);
  assert.ok(clock.equal_reads > 2000);
  assert.throws(() => clockProbe(() => 0), /never advanced/);
  assert.throws(() => clockProbe(() => -reads++), /invalid probe/);
});
