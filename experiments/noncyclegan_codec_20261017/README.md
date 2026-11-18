# Non-CycleGAN codec pilot, 2026-10-17

Read report.md for completed results and research.md for primary literature and limitations. These are our adaptations, not paper reproductions or speaker-verification results.

Runtime: existing Python 3.8.10 / torch 2.4.1+cu121 / compatible torchaudio, numpy 1.24, scipy, matplotlib. Isolated added packages: transformers 4.46.3, huggingface_hub 0.27.1, tokenizers 0.20.3, safetensors 0.5.3, regex 2024.11.6. No base environment upgrade.

Put official pinned kyutai/mimi config.json and model.safetensors in mimi/ (revision and SHA256 in model_provenance.json). Run `python -u run_codec_adapter.py` on CUDA. It executes finite fits and writes checkpoints, histories, results.json and validation plots. The entire codec stays frozen; only small ACC adapters train.

Private inputs are intentionally not included: pairs_train.npz and pairs_val.npz must contain x=(N,80,81) original normalized ACC, audio16=(N,12800), audio24=(N,19200), and recording ids. N=58 training windows/20 recordings, N=16 validation windows/4 other recordings. Waveforms must use the same exact original nominal window starts; audio24 is resample_poly(audio16,3,2), without per-record gain. Put original reference audio Mel arrays in ../data_original/{train,val}/imu_audio.npy. Script checks recomputed waveform Mel against these references. Fixed scales belong to this frozen experiment only; do not reuse them blindly on different sensors or recalibrate individually by user.

The helper filename data0723_frontend_helpers.py is historical: only its generic original Mel filterbank is imported; no Data0723 arrays or users are loaded by this experiment. Current source is the same person's vr_imu_collector raw IMU recordings. Test and authentication-reserve recordings are excluded.

No raw waveforms, private arrays, large pretrained weights or authentication reserve are published. Model quality failure is preserved rather than reported as improved authentication.
