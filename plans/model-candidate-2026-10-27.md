# Focused candidate release evaluation — 2026-10-27

Candidate SHA-256: `0e565a6a4a856068388c0a4b7f5b1bcc2192a9694f51a2e9a97783ea1d8cb921`  
R20 SHA-256: `cd561330a1666ed3286f3f411277144c335f096b4105190864b5adc49054f135`

The candidate passes the documented promotion gate: independent native and production-browser paired lower 95% bounds exceed 52%, and no enumerated external matchup shows a statistically supported regression. This supports publishing it as stronger overall than R20 under the two independent project strength gates.

| Gate | Candidate result | 95% interval | Outcome |
|---|---:|---:|---|
| Native vs R20, 1,024 opening pairs / 2,048 games | 684–865–499 W/D/L; 54.516% | paired 52.929–56.104% | Pass |
| Production browser vs R20, 1,024 pairs / 2,048 games | 647–1,002–399 W/D/L; 56.055% | paired 54.558–57.552%; opening-cluster normal 54.111–57.998%; cluster bootstrap 54.114–57.989% | Pass |

The browser confirmation used the pinned production worker and model hashes, headless Chromium 156, and Hard settings (4,096 simulation cap, 140 ms, tactical proof). The model artifact is 1,186,321 bytes, below the 1,500,000-byte limit.

| Fixed external check | Candidate vs R20 score difference | Paired 95% interval | Outcome |
|---|---:|---:|---|
| Rust alpha-beta | −1.953 pp | −5.36 to +1.45 pp | Inconclusive |
| Rust learned PUCT | −1.855 pp | −3.826 to +0.115 pp | Inconclusive |
| uttt.ai | −1.807 pp | −4.952 to +1.339 pp | Inconclusive |
| CodinGame | −0.781 pp | −1.888 to +0.326 pp | Inconclusive |

The fixed offline heuristic benchmark scored 4,851 wins, 13 draws, and 0 losses in 4,864 games (99.867% score). It did not meet the 100% target, and R20's historical 99.959% score was slightly higher. Significant improvement against the enumerated external opponents was also not established; all four confidence intervals include zero. These stretch goals remain open despite passing the documented release gate.
