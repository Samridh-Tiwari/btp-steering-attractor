# Steering an attractor

**What latent dynamics models can and cannot tell us about controlling a neural circuit**

B.Tech Project (PH587), Department of Physics, IIT Bombay, 2026.
Samridh Tiwari, guided by Prof. Amitabha Nandi.

Report: [`report/Steering_an_attractor_BTP_final_Samridh_Tiwari.pdf`](report/Steering_an_attractor_BTP_final_Samridh_Tiwari.pdf)

## The question

During a summer internship we fitted LFADS to mouse motor cortex. The fitted model looked like a continuous attractor holding the plan for a lever pull, but an optimal (LQR) controller pushing through the model's four inferred input channels removed only 0.1% of the error. The explanation at the time was that four channels are too few for a 128-dimensional state.

This project tests that explanation where the truth is known. A recurrent "teacher" network trained to load, hold and release a plan plays the role of the circuit. Models are fitted only to Poisson spikes generated from it: a simplified LFADS-style student, and full LFADS built from the [lfads-torch](https://github.com/arsedler9/lfads-torch) modules. The main steering result is repeated on the public MC_Maze dataset (monkey motor cortex). No data from the internship laboratory are used anywhere in this repository.

## Main results

- **Channel count is not the limit.** On the teacher a single, well aimed channel has 100% of the authority of pushing on all 100 units. The best push is the left eigenvector of the slow mode, 24° from the line of states, and plan decoders do not find it.
- **Inferred channels are poorly aimed.** The simplified student's four channels reach 15–51% of full authority (best four: 56–100%). Full LFADS fits the spikes far better (single-trial rate R² 0.94–0.96) but its channels reach only 0.04–0.8%, below random (6%). LQR through them removes 0.4–6% of the error at a fair effort cost. On MC_Maze the figure is 0.3–0.5% against 100% for the best four.
- **Why: the gated generator.** A GRU unit holds a value by closing its update gate, and the same gate shuts out its input. In the delay, units with closed gates carry 74–98% of the behavioural sensitivity but receive at most 3% of the controller's push.
- **Fitted models do not keep the plan as the circuit does.** The simplified student is handed the plan by its controller at the go cue. Full LFADS writes the plan into its initial state before the plan cue (R² 0.96–0.99 on held-out trials), and so reports an attractor even for a teacher with no memory.
- **The reported mechanism follows the model's assumptions.** Held-out prediction and agreement between seeds do not choose between them; a perturbation does.

## Repository layout

```
code/      all analysis code (run every script from inside code/)
code/runs/ summary results as JSON, the three trained teacher networks (*.pt), and
           per_fit_results.zip (one result file per fitted student; unzip it here
           before re-making the figures)
code/figs/ figures produced by the scripts
report/    LaTeX source, figures and compiled PDF of the report
```

Main files in `code/`:

| File | What it does |
| --- | --- |
| `teacher.py`, `model.py`, `tasks.py`, `train.py` | the teacher network, its task and training |
| `make_data.py`, `split_study.py`, `make_noisy.py` | spikes from teachers A and A2, teacher B (no memory), and teacher A with dynamics noise |
| `lfads_lite.py` | the simplified LFADS student |
| `event_student.py` | the "told-when" student (inputs only at the cue times) |
| `lfads_full.py` | full LFADS from the lfads-torch modules, own training loop |
| `control.py`, `steer.py`, `teacher_steer.py`, `student_steer.py` | behavioural authority and the LQR link |
| `plan_tests.py`, `event_tests.py`, `fp_count.py`, `lfads_tests.py` | swap, shift, persistence, fixed points, transfer |
| `gate_check.py`, `v_timecourse.py`, `ensemble.py`, `rate_quality.py` | GRU gates, when the plan is known, agreement between fits, rate accuracy |
| `mc_maze.py`, `mc_plan_tests.py`, `lfads_mc_tests.py` | the same analyses on MC_Maze |
| `fig*.py`, `diagrams*.py`, `figstyle.py` | all figures and schematics |
| `aggregate.py` | collects every number quoted in the report into `runs/summary.json` |

## Running it

Python 3.10+, CPU only. Install the requirements, and for full LFADS clone lfads-torch next to this repository (or set `LFADS_TORCH_DIR`); only its `modules` are used, not its training framework.

```bash
pip install -r requirements.txt
git clone https://github.com/arsedler9/lfads-torch ../lfads-torch
cd code

python make_data.py                                   # teacher A and A2 spikes (teachers are in runs/)
python split_study.py data_b                           # teacher B spikes
python make_noisy.py                                   # teacher A with dynamics noise

python lfads_lite.py --teacher x --k 4 --lam_u 0.1 --seed 0          # one simplified student
python event_student.py --data runs/dataset.pt --steps 7000 --seed 0 # one told-when student
python lfads_full.py --data runs/dataset.pt --tag A --kl_co 1e-4 --seed 0 --n_train 192   # one full LFADS fit

python teacher_steer.py && python student_steer.py     # steering authority
python plan_tests.py A B T2 && python event_tests.py   # where the plan is kept
python lfads_tests.py runs/lfads_A_kl0.0001_s0_tr192.pt
```

To retrain the teachers from scratch: `python teacher.py --g 0.9 --noise 0.05 --seed 0` (and `--seed 1`), and `python split_study.py train_b`.

MC_Maze: download the NWB file from [DANDI 000128](https://dandiarchive.org/dandiset/000128) (CC-BY-4.0), then `python mc_maze.py prep <file.nwb>`, `python mc_maze.py fit --k 4 --lam_u 0.1 --seed 0` and `python mc_maze.py steer`.

A simplified student takes 2–7 minutes on one CPU core, a told-when student about 8 and a full LFADS fit 20–40. The report used about 150 fits.

## Data and credits

- MC_Maze: Churchland and Kaufman, via the Neural Latents Benchmark (Pei et al. 2021), DANDI 000128.
- LFADS modules: Sedler and Pandarinath, lfads-torch (2023).
