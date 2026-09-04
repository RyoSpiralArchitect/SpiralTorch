"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const crypto = require("node:crypto");
const { preflight, collectPayload, finalize } = require("../scripts/finalize_github_release.cjs");

function fixture(t) {
  const dist = fs.mkdtempSync(path.join(os.tmpdir(), "release-contract-"));
  t.after(() => fs.rmSync(dist, { recursive: true, force: true }));
  const lines = [];
  for (const name of ["linux.whl", "mac.whl", "win.whl"]) {
    fs.writeFileSync(path.join(dist, name), name);
    fs.writeFileSync(path.join(dist, name + ".sigstore.json"), "{}");
    lines.push(crypto.createHash("sha256").update(name).digest("hex") + "  " + name);
  }
  fs.writeFileSync(path.join(dist, "wheels.sha256"), lines.join("\n") + "\n");
  fs.writeFileSync(path.join(dist, "wheels.sha256.sigstore.json"), "{}");
  return dist;
}

function client() {
  const state = { release: null, assets: [], events: [], sha: "commit" };
  const repos = {
    getReleaseByTag: async () => {
      if (state.readError) throw state.readError;
      if (!state.release) throw Object.assign(new Error("missing"), { status: 404 });
      return { data: state.release };
    },
    getCommit: async (args) => {
      assert.equal(args.ref, "refs/tags/v1");
      return { data: { sha: state.sha } };
    },
    createRelease: async (args) => {
      state.events.push(["create", args]);
      state.release = { ...args, id: 1, immutable: false };
      return { data: state.release };
    },
    listReleaseAssets: () => {},
    uploadReleaseAsset: async (args) => {
      state.events.push(["upload", args.name]);
      if (state.uploadError) throw new Error("uncertain upload");
      state.assets.push({
        name: args.name, size: args.data.length, state: "uploaded",
        digest: "sha256:" + crypto.createHash("sha256").update(args.data).digest("hex"),
      });
      if (state.moveTag) state.sha = "other";
      if (state.corruptAsset) state.assets.at(-1).digest = "sha256:bad";
      return { data: state.assets.at(-1) };
    },
    updateRelease: async (args) => {
      state.events.push(["publish", args]);
      if (state.publishError) throw new Error("uncertain publish");
      state.release = { ...state.release, ...args };
      return { data: state.release };
    },
  };
  const github = { rest: { repos }, paginate: async () => state.assets };
  return { state, github, repo: { owner: "test", repo: "repo" }, tag: "v1", sourceSha: "commit" };
}

test("new release stays draft until every asset matches", async (t) => {
  const c = client();
  const result = await finalize({ ...c, dist: fixture(t) });
  assert.equal(result.draft, false);
  assert.equal(c.state.events[0][0], "create");
  assert.equal(c.state.events[0][1].draft, true);
  assert.equal(c.state.events.filter(([op]) => op === "upload").length, 8);
  assert.deepEqual(c.state.events.at(-1), [
    "publish", { ...c.repo, release_id: 1, draft: false },
  ]);
});

test("published or immutable releases are rejected without writes", async () => {
  for (const release of [
    { draft: false, immutable: true },
    { draft: false, immutable: false },
    { draft: true, immutable: true },
    { draft: true, immutable: false, tag_name: "other" },
  ]) {
    const c = client();
    c.state.release = { tag_name: c.tag, ...release };
    await assert.rejects(preflight(c), /Refusing/);
    assert.deepEqual(c.state.events, []);
  }
});

test("non-404 lookup errors are not interpreted as missing", async () => {
  const c = client();
  c.state.readError = Object.assign(new Error("denied"), { status: 403 });
  await assert.rejects(preflight(c), /denied/);
  assert.deepEqual(c.state.events, []);
});

test("an exact partial draft resumes without overwriting matching assets", async (t) => {
  const c = client();
  const dist = fixture(t);
  const files = collectPayload(dist);
  const [name, file] = [...files][0];
  c.state.release = { id: 2, tag_name: c.tag, draft: true, immutable: false };
  c.state.assets = [{ name, size: file.size, digest: file.digest, state: "uploaded" }];
  await finalize({ ...c, dist });
  assert.equal(c.state.events.filter(([op]) => op === "upload").length, files.size - 1);
  assert.equal(c.state.events.some(([op]) => op === "create"), false);
  assert.equal(c.state.events.some(([op, arg]) => op === "upload" && arg === name), false);
});

for (const change of ["digest", "size", "name", "state", "duplicate"]) {
  test("conflicting draft asset blocks writes: " + change, async (t) => {
    const c = client();
    const dist = fixture(t);
    const [name, file] = [...collectPayload(dist)][0];
    const asset = { name, size: file.size, digest: file.digest, state: "uploaded" };
    if (change !== "duplicate") asset[change] = "wrong";
    c.state.release = { id: 2, tag_name: c.tag, draft: true };
    c.state.assets = change === "duplicate" ? [asset, asset] : [asset];
    await assert.rejects(finalize({ ...c, dist }), /Remote asset mismatch/);
    assert.deepEqual(c.state.events, []);
  });
}

for (const fault of ["uploadError", "corruptAsset", "moveTag"]) {
  test("staging fault never publishes: " + fault, async (t) => {
    const c = client();
    c.state[fault] = true;
    await assert.rejects(finalize({ ...c, dist: fixture(t) }));
    assert.equal(c.state.events.some(([op]) => op === "publish"), false);
    if (fault === "uploadError") {
      assert.equal(c.state.events.filter(([op]) => op === "upload").length, 1);
    }
  });
}

test("uncertain publication is not retried", async (t) => {
  const c = client();
  c.state.publishError = true;
  await assert.rejects(finalize({ ...c, dist: fixture(t) }), /uncertain publish/);
  assert.equal(c.state.events.filter(([op]) => op === "publish").length, 1);
});

test("wrong tag source prevents draft creation", async (t) => {
  const c = client();
  c.state.sha = "other";
  await assert.rejects(finalize({ ...c, dist: fixture(t) }), /Release tag/);
  assert.deepEqual(c.state.events, []);
});

for (const fault of ["checksum", "bundle", "duplicate", "symlink", "extra-wheel"]) {
  test("invalid local payload prevents all remote writes: " + fault, async (t) => {
    const c = client();
    const dist = fixture(t);
    if (fault === "checksum") fs.appendFileSync(path.join(dist, "linux.whl"), "changed");
    if (fault === "bundle") fs.unlinkSync(path.join(dist, "linux.whl.sigstore.json"));
    if (fault === "duplicate") {
      fs.mkdirSync(path.join(dist, "nested"));
      fs.writeFileSync(path.join(dist, "nested/linux.whl"), "linux.whl");
    }
    if (fault === "symlink") fs.symlinkSync("linux.whl", path.join(dist, "alias"));
    if (fault === "extra-wheel") fs.writeFileSync(path.join(dist, "extra.whl"), "extra");
    await assert.rejects(finalize({ ...c, dist }));
    assert.deepEqual(c.state.events, []);
  });
}
