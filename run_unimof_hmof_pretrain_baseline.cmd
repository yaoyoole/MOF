@echo off
setlocal EnableExtensions

REM Official Uni-MOF hMOF baseline, Windows / single RTX 5090 / FP32.
REM Run this from an Anaconda Prompt after: conda activate unimof_clean
REM This launcher deliberately does not alter Uni-MOF or Uni-Core source code.
REM Usage: no argument runs the planned 10-epoch review stage.
REM        pass 50 after review to resume checkpoint_last.pt through epoch 50.

set "PROJECT_ROOT=D:\pycharm\Py_projects\GNN"
set "UNIMOF_ROOT=%PROJECT_ROOT%\external\Uni-MOF"
set "DATA_ROOT=%PROJECT_ROOT%\data\processed\unimof_hmof_lmdb"
set "SAVE_DIR=%PROJECT_ROOT%\results\unimof_hmof_pretrain_baseline"
set "TARGET_EPOCH=%~1"
if "%TARGET_EPOCH%"=="" set "TARGET_EPOCH=10"

if not "%TARGET_EPOCH%"=="10" if not "%TARGET_EPOCH%"=="20" if not "%TARGET_EPOCH%"=="30" if not "%TARGET_EPOCH%"=="40" if not "%TARGET_EPOCH%"=="50" (
    echo ERROR: pass 10, 20, 30, 40, or 50 as the target epoch.
    exit /b 1
)

if not exist "%DATA_ROOT%\train.lmdb" (
    echo ERROR: train.lmdb was not found at "%DATA_ROOT%".
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found. Activate unimof_clean first.
    exit /b 1
)

if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"
pushd "%UNIMOF_ROOT%"

REM 110122 train structures / effective batch 32 = ceil(3441.3125) = 3442 updates/epoch.
REM The scheduler always spans 50 epochs = 172100 updates; warmup is 10%% = 17210.
REM --max-epoch controls only where this invocation pauses.  Thus the first
REM 10-epoch stage and a later 50-epoch resume share one continuous LR schedule.
REM Every 3442 updates: validate and save. keep-interval-updates=1 retains only
REM the most recent periodic checkpoint, in addition to checkpoint_last/best.
echo Uni-MOF hMOF baseline: running through epoch %TARGET_EPOCH% of planned 50.
python "%PROJECT_ROOT%\scripts\tee_process_output.py" --log "%SAVE_DIR%\training_stage_%TARGET_EPOCH%epochs.log" -- ^
  python -m unicore_cli.train "%DATA_ROOT%" ^
  --user-dir .\unimof ^
  --train-subset train ^
  --valid-subset valid ^
  --num-workers 0 ^
  --distributed-world-size 1 ^
  --task unimat ^
  --loss unimat ^
  --arch unimat_base ^
  --optimizer adam ^
  --adam-betas "(0.9,0.99)" ^
  --adam-eps 1e-6 ^
  --clip-norm 1.0 ^
  --weight-decay 1e-4 ^
  --lr-scheduler polynomial_decay ^
  --lr 3e-4 ^
  --warmup-updates 17210 ^
  --total-num-update 172100 ^
  --max-update 172100 ^
  --max-epoch %TARGET_EPOCH% ^
  --batch-size 4 ^
  --batch-size-valid 4 ^
  --update-freq 8 ^
  --seed 1 ^
  --log-interval 25 ^
  --log-format simple ^
  --save-interval-updates 3442 ^
  --validate-interval-updates 3442 ^
  --keep-interval-updates 1 ^
  --best-checkpoint-metric loss ^
  --no-epoch-checkpoints ^
  --tensorboard-logdir "%SAVE_DIR%\tsb" ^
  --masked-token-loss 1 ^
  --masked-coord-loss 1 ^
  --masked-dist-loss 1 ^
  --lattice-loss 1 ^
  --x-norm-loss 0.01 ^
  --delta-pair-repr-norm-loss 0.01 ^
  --mask-prob 0.15 ^
  --noise-type uniform ^
  --noise 1.0 ^
  --dist-threshold 5.0 ^
  --minkowski-p 2.0 ^
  --required-batch-size-multiple 1 ^
  --remove-hydrogen ^
  --save-dir "%SAVE_DIR%"

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
