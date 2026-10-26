#!/usr/bin/env -S npx tsx
// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * aamt-model — bake a safetensors checkpoint into a GPU-native `.aamt`, or inspect one.
 *
 *   aamt-model bake <in.safetensors> <out.aamt> [--kind int4|ternary|1bit] [--group 32]
 *   aamt-model inspect <file.aamt>
 */
import { statSync } from 'node:fs';
import { resolve } from 'node:path';
import { inspectAamt } from './amdl-inspect';
import { bakeSafetensorsFile } from './bake';
import type { AamtKind } from './tensor';

const [cmd, ...rest] = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : undefined;
};
// `pnpm bake` runs this from the package directory; INIT_CWD is where the user typed the command.
const here = process.env.INIT_CWD ?? process.cwd();
const positional = rest.filter((a, i) => !a.startsWith('--') && !rest[i - 1]?.startsWith('--')).map((p) => resolve(here, p));
const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

if (cmd === 'bake' && positional.length === 2) {
  const [input, output] = positional as [string, string];
  const kind = (flag('kind') ?? 'int4') as AamtKind;
  if (!['int4', 'ternary', '1bit'].includes(kind)) throw new Error(`unknown --kind ${kind}`);
  const groupSize = Number(flag('group') ?? 32);
  const t0 = performance.now();
  const s = bakeSafetensorsFile(input, output, {
    kind,
    groupSize,
    onTensor: (e, i, n) => {
      if (process.stdout.isTTY) process.stdout.write(`\r  [${i + 1}/${n}] ${e.kind?.padEnd(7)} ${e.name.slice(0, 60).padEnd(60)}`);
    },
  });
  const kinds = Object.entries(s.kinds).map(([k, n]) => `${n} ${k}`).join(', ');
  console.log(`${process.stdout.isTTY ? '\n' : ''}baked ${s.tensors} tensors (${kinds}) in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  console.log(`${mb(s.bytesIn)} -> ${mb(s.bytesOut)}  (${((s.bytesOut / s.bytesIn) * 100).toFixed(1)}% of the input)`);
} else if (cmd === 'inspect' && positional.length === 1) {
  const r = await inspectAamt(positional[0]!);
  if (!r.ok) {
    console.error(`not an .aamt model: ${r.error}`);
    process.exit(1);
  }
  console.log(`file:      ${r.path} (${mb(statSync(r.path).size)})`);
  console.log(`format:    AMDL v${r.footer?.version}, ${r.tensorCount} tensors`);
  console.log(`kinds:     ${Object.entries(r.kinds ?? {}).map(([k, n]) => `${k} ${n}`).join(' · ')}`);
  if (r.family) console.log(`family:    ${r.family}`);
} else {
  console.error('usage:\n  aamt-model bake <in.safetensors> <out.aamt> [--kind int4|ternary|1bit] [--group 32]\n  aamt-model inspect <file.aamt>');
  process.exit(2);
}
