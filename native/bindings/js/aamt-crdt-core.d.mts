// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Types for aamt-crdt-core.mjs, the browser-safe CRDT core (see that file for the rules).
export type Tera = [number, number, number, number];

export declare class HLC {
  readonly physical: number;
  readonly logical: number;
  readonly replicaId: bigint;
  constructor(physical: number, logical: number, replicaId: bigint | number | string);
  static zero(replicaId: bigint | number | string): HLC;
  tick(nowMs?: number): HLC;
  update(remote: HLC, nowMs?: number): HLC;
  compare(other: HLC): -1 | 0 | 1;
  toJSON(): { physical: number; logical: number; replica_id: string };
  static fromJSON(value: { physical: number; logical: number; replica_id: string }): HLC;
}

export interface CrdtRecord {
  id: string;
  tera: Tera;
  data: Uint8Array;
  hlc: HLC;
}

export declare class CrdtStore {
  readonly records: Map<string, CrdtRecord>;
  constructor(records?: Map<string, CrdtRecord>);
  /** True when the record won (new key, or a greater (hlc, data, tera) than the one held). */
  put(id: string | number, tera: readonly number[], data: Uint8Array, hlc: HLC): boolean;
  merge(other: CrdtStore): CrdtStore;
  get(id: string | number): CrdtRecord | undefined;
  readonly size: number;
  values(): CrdtRecord[];
  toJSON(): { records: { id: string | number; tera: number[]; data: string; hlc: ReturnType<HLC['toJSON']> }[] };
  toJson(): string;
  static fromJSON(value: string | ReturnType<CrdtStore['toJSON']>): CrdtStore;
}

export declare function b64encode(bytes: Uint8Array): string;
export declare function b64decode(text: string): Uint8Array;
export declare function big(value: bigint | number | string): bigint;
export declare function randomReplicaId(): bigint;
