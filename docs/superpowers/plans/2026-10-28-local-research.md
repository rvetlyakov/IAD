# Local assignment GA research

User specification: continue on main locally; independent TRAIN/SURVEY/HOLDOUT;
no fabricated points; n<100, 100–499, 500–2000; equal N/G/local budgets.

- [x] Compile deterministic swap/cycle loops with optional Numba; compare against Python reference on random and diagonal matrices.
- [x] Reuse a process executor across runs; record actual worker PIDs and migration events. Test ring migration independently.
- [x] Run a bounded, documented coordinate TRAIN search for every active factor, including island count. Freeze settings before SURVEY/HOLDOUT.
- [x] Strict OFAT over a dense 10–2000 size grid; save every completed run immediately with settings, seeds, exact cost, time, budget, permutation validity.
- [x] Independent paired holdout with equal total population, generations, local calls and local candidate/pass settings; diagonal checks.
- [x] PNG/SVG plots without fitted observations, parameter tables, conclusions and manifest.
- [x] Ruff, compileall, entire unittest suite; inspect plots; synchronize GitHub with measured evidence.

Resource choice: JIT only deterministic local loops, warm before timings;
fixed small search grids instead of unbounded Cartesian tuning. Best means best
observed in this declared search, never global optimum. Persistent executor
startup reported separately from warm steady-state algorithm time.
