# Public demo deployment

The app requires Python hosting because TensorFlow inference runs on the server. Static GitHub Pages and Cloudflare Workers do not run this Python application.

## Streamlit Community Cloud

1. Sign in to https://share.streamlit.io/ and accept its terms yourself.
2. Create an app from `amirhamzaaaa4/cifar-lens`, branch `main`, entrypoint `app.py`.
3. Select Python 3.12 in advanced settings.
4. Deploy after the trained experiment has been included. The app needs `model.keras`, `results.json`, `errors.npz` and optionally the `samples` folder under `artifacts/<experiment>`.
5. Test a sample and an uploaded PNG, check scores and Grad-CAM, then copy the actual deployment URL into README. Never substitute a guessed URL.

## Container hosting

A Dockerfile is included for any compatible Python container host. Supply the experiment artifacts before building. Model files are intentionally ignored by regular Git tracking; publish deployment copies explicitly after the benchmark has been verified. Keep interim training checkpoints out of the public runtime.

Uploads are processed in memory and are not saved by the application. Visitors should use ordinary object photos suitable for public demo processing.
