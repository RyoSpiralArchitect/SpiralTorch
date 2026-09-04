"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const crypto = require("node:crypto");
const { preflight, collectPayload, recoveryArtifact, finalize } = require("../scripts/finalize_github_release.cjs");

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
      if (!state.release || state.release.draft) {
        throw Object.assign(new Error("tag endpoint does not expose drafts"), { status: 404 });
      }
      return { data: state.release };
    },
    listReleases: () => {},
    getRelease: async (args) => {
      assert.equal(args.release_id, state.release.id);
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
  const github = { rest: { repos }, paginate: async (method) => (
    method === repos.listReleases ? (state.release ? [state.release] : []) : state.assets
  ) };
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
  ]) {
    const c = client();
    c.state.release = { tag_name: c.tag, ...release };
    await assert.rejects(preflight(c), /Refusing/);
    assert.deepEqual(c.state.events, []);
  }
});

test("authenticated pagination discovers a draft hidden by the tag endpoint", async () => {
  const c = client();
  c.state.release = { id: 9, tag_name: c.tag, draft: true };
  assert.equal((await preflight(c)).id, 9);
  assert.deepEqual(c.state.events, []);
});

test("ambiguous draft lookup is rejected instead of creating another release", async () => {
  const c = client();
  c.state.release = { id: 9, tag_name: c.tag, draft: true };
  c.github.paginate = async () => [c.state.release, { ...c.state.release, id: 10 }];
  await assert.rejects(preflight(c), /Ambiguous/);
  assert.deepEqual(c.state.events, []);
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

test("explicit recovery uses retained bytes without reuploading or resigning", async (t) => {
  const c = client();
  const dist = fixture(t);
  c.state.publishError = true;
  await assert.rejects(finalize({ ...c, dist }), /uncertain publish/);
  const uploads = c.state.events.filter(([op]) => op === "upload").length;
  c.state.publishError = false;
  await finalize({ ...c, dist });
  assert.equal(c.state.events.filter(([op]) => op === "upload").length, uploads);
  assert.equal(c.state.release.draft, false);
});

function recoveryClient() {
  const run = {
    path: ".github/workflows/release_wheels.yml", event: "push",
    head_branch: "v1", head_sha: "commit", status: "completed", conclusion: "failure",
  };
  const artifact = {
    id: 7, name: "signed-release-payload-v1", expired: false,
    digest: "sha256:" + "a".repeat(64),
  };
  return {
    run, artifact, repo: { owner: "test", repo: "repo" },
    tag: "v1", sourceSha: "commit", runId: "123",
    github: {
      rest: { actions: {
        getWorkflowRun: async () => ({ data: run }), listWorkflowRunArtifacts: () => {},
      } },
      paginate: async () => [artifact],
    },
  };
}

test("recovery selects a retained artifact from a failed exact-tag build", async () => {
  const c = recoveryClient();
  assert.equal((await recoveryArtifact(c)).id, 7);
});

for (const field of ["path", "event", "head_branch", "head_sha", "status"]) {
  test("recovery rejects mismatched build identity: " + field, async () => {
    const c = recoveryClient();
    c.run[field] = "wrong";
    await assert.rejects(recoveryArtifact(c), /exact source/);
  });
}

for (const field of ["name", "expired", "digest"]) {
  test("recovery rejects invalid retained artifact: " + field, async () => {
    const c = recoveryClient();
    c.artifact[field] = field === "expired" ? true : "wrong";
    await assert.rejects(recoveryArtifact(c), /retained/);
  });
}

test("recovery rejects missing/duplicate artifacts and malformed run ids", async () => {
  const c = recoveryClient();
  for (const artifacts of [[], [c.artifact, c.artifact]]) {
    c.github.paginate = async () => artifacts;
    await assert.rejects(recoveryArtifact(c), /retained/);
  }
  for (const runId of ["0", "-1", "../other", "9007199254740992"]) {
    await assert.rejects(recoveryArtifact({ ...c, runId }), /run id/);
  }
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
