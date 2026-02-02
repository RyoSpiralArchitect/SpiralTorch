import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";

type SpiralWasmExampleConfig = {
  port: number;
};

function ensureDir(dir: string) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyDirRecursive(srcDir: string, destDir: string) {
  ensureDir(destDir);
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);

    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
      continue;
    }

    if (entry.isSymbolicLink()) {
      const target = fs.readlinkSync(srcPath);
      try {
        fs.unlinkSync(destPath);
      } catch {
      }
      fs.symlinkSync(target, destPath);
      continue;
    }

    if (entry.isFile()) {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function exampleHeaders() {
  return {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
  } as const;
}

export function defineSpiralWasmExampleConfig(options: SpiralWasmExampleConfig) {
  return defineConfig({
    root: ".",
    base: "./",
    build: {
      outDir: "dist",
      emptyOutDir: true,
    },
    server: {
      port: options.port,
      headers: exampleHeaders(),
    },
    preview: {
      port: options.port,
      headers: exampleHeaders(),
    },
    plugins: [
      {
        name: "spiraltorch-copy-wasm-pkg",
        apply: "build",
        closeBundle() {
          const pkgDir = path.resolve(process.cwd(), "pkg");
          const distPkgDir = path.resolve(process.cwd(), "dist", "pkg");

          if (!fs.existsSync(pkgDir)) {
            console.warn(
              `[spiraltorch] missing ./pkg. Run ./scripts/build_wasm_web.sh --dev (from repo root) first.`,
            );
            return;
          }

          copyDirRecursive(pkgDir, distPkgDir);
        },
      },
    ],
  });
}
