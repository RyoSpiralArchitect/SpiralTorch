"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

function assertDraft(release, tag) {
  if (release.tag_name !== tag || release.draft !== true || release.immutable === true) {
    throw new Error("Refusing to mutate a published or mismatched release; use a new version.");
  }
}

async function getRelease(github, repo, tag) {
  try {
    return (await github.rest.repos.getReleaseByTag({ ...repo, tag })).data;
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}

async function preflight({ github, repo, tag }) {
  const release = await getRelease(github, repo, tag);
  if (release) assertDraft(release, tag);
  return release;
}

function collectPayload(dist, expectedWheels = 3) {
  const files = new Map();
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const filename = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(filename);
      } else {
        if (!entry.isFile() || files.has(entry.name)) {
          throw new Error("Payload contains a symlink, special file, or duplicate basename.");
        }
        const bytes = fs.readFileSync(filename);
        files.set(entry.name, {
          filename,
          size: bytes.length,
          digest: "sha256:" + crypto.createHash("sha256").update(bytes).digest("hex"),
        });
      }
    }
  }
  visit(dist);
  const manifest = files.get("wheels.sha256");
  if (!manifest) throw new Error("Missing wheels.sha256.");
  const wheels = new Map([...files].filter(([name]) => name.endsWith(".whl")));
  const seen = new Set();
  for (const line of fs.readFileSync(manifest.filename, "utf8").trim().split(/\r?\n/)) {
    const match = /^([a-f0-9]{64})  (\S+\.whl)$/.exec(line);
    if (!match || seen.has(match[2]) || wheels.get(match[2])?.digest !== "sha256:" + match[1]) {
      throw new Error("Wheel checksum manifest does not match the payload.");
    }
    seen.add(match[2]);
  }
  if (seen.size !== expectedWheels || wheels.size !== expectedWheels) {
    throw new Error("Unexpected release wheel count.");
  }
  for (const name of files.keys()) {
    if (!name.endsWith(".sigstore.json") && !files.has(name + ".sigstore.json")) {
      throw new Error("Missing Sigstore bundle for " + name);
    }
  }
  return files;
}

function assertAssets(assets, files, complete) {
  const seen = new Set();
  for (const asset of assets) {
    const expected = files.get(asset.name);
    if (!expected || seen.has(asset.name) || asset.state !== "uploaded" ||
        asset.size !== expected.size || asset.digest !== expected.digest) {
      throw new Error("Remote asset mismatch; refusing to replace or publish: " + asset.name);
    }
    seen.add(asset.name);
  }
  if (complete && seen.size !== files.size) throw new Error("Release assets are incomplete.");
  return seen;
}

async function finalize({ github, repo, tag, sourceSha, dist }) {
  // Validate locally before making any remote mutation.
  const files = collectPayload(dist);
  const checkSource = async () => {
    const { data } = await github.rest.repos.getCommit({ ...repo, ref: "refs/tags/" + tag });
    if (data.sha !== sourceSha) throw new Error("Release tag changed or does not match the build.");
  };
  await checkSource();
  let release = await preflight({ github, repo, tag });
  if (!release) {
    release = (await github.rest.repos.createRelease({
      ...repo, tag_name: tag, target_commitish: sourceSha, draft: true,
      name: "SpiralTorch " + tag,
      body: "Signed release wheels built from " + sourceSha + ".",
    })).data;
  }
  assertDraft(release, tag);
  const listAssets = () => github.paginate(github.rest.repos.listReleaseAssets, {
    ...repo, release_id: release.id, per_page: 100,
  });
  const existing = assertAssets(await listAssets(), files, false);
  for (const [name, file] of files) {
    if (existing.has(name)) continue;
    await github.rest.repos.uploadReleaseAsset({
      ...repo, release_id: release.id, name,
      headers: { "content-type": "application/octet-stream" },
      data: fs.readFileSync(file.filename),
    });
  }
  assertAssets(await listAssets(), files, true);
  const latest = await getRelease(github, repo, tag);
  if (!latest || latest.id !== release.id) throw new Error("Release identity changed.");
  assertDraft(latest, tag);
  await checkSource();
  // Publish last, without changing target_commitish or retrying an ambiguous write.
  return (await github.rest.repos.updateRelease({
    ...repo, release_id: release.id, draft: false,
  })).data;
}

module.exports = { assertDraft, assertAssets, collectPayload, preflight, finalize };
