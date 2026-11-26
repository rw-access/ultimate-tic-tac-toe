import { createHash } from 'node:crypto';

export type OpeningSchedule = 'legacy' | 'independent';

export function parseOpeningSchedule(value?: string): OpeningSchedule {
  if (value === undefined || value === 'legacy') return 'legacy';
  if (value === 'independent') return value;
  throw new Error('--opening-schedule must be legacy or independent');
}

/** Independent seeds are the first big-endian u32 of a domain-separated UTF-8 SHA-256.
 * Pair indices start at zero and remain absolute when running a shard of the schedule. */
export function pairSeedGenerator(schedule: OpeningSchedule, matchSeed: number, legacyNext: () => number): (pair: number) => number {
  if (schedule === 'legacy') return legacyNext;
  return pair => createHash('sha256').update(`uttt.arena.openings.v1\0${matchSeed}\0${pair}`, 'utf8').digest().readUInt32BE(0);
}
