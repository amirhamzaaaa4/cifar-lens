# Public demo deployment

The live deployment is now the free [Hugging Face static Space](https://huggingface.co/spaces/AMIRHAMZAAAA4/cifar-lens). Upload the contents of `web/` directly to a Static Space. It requires no GitHub or Drive authorization. The model runs in the visitor's browser; the host serves public files. Browser and Python predictions are compared on thirty identical CIFAR-10 images before inference is enabled.

Rebuild the browser assets from a verified full experiment with `python -m lens.export_browser --experiment artifacts/resnet20-corrected-s73 --output web`. This exports the layer graph, float32 weights with checksum, metrics, real error images and Python reference logits/probabilities. The static `README.md`, HTML, CSS and JavaScript are maintained alongside them.

Hugging Face's account setup currently requires a paid plan for Docker/Gradio compute. The following alternatives describe running the full Python Streamlit application, not the free static demo.

TensorFlow inference runs on the server, so the app needs Python hosting. Static hosts such as GitHub Pages cannot run it.

## 1. Publish the trained model

1. Finish a full training run (the Colab notebook's last cell builds a release zip without training checkpoints).
2. On GitHub: **Releases → Draft a new release** (e.g. tag `v1.0`), attach the zip and publish.
3. Copy the asset URL, e.g. `https://github.com/amirhamzaaaa4/cifar-lens/releases/download/v1.0/resnet20-s73.zip`.

The app's `lens.fetch` verifies the model's SHA-256 against `results.json` before using it.

## 2a. Hugging Face Spaces (recommended)

1. Create a Space with the **Docker** SDK and push this repository to it (or link it from GitHub).
2. In the Space settings, add the variable `LENS_EXPERIMENT_URL` = your release asset URL.
3. HF Spaces expects port 7860 by default: either set `app_port: 8501` in the Space README front matter, or change the port in the Dockerfile `CMD`.
4. Open the Space, test a sample and an uploaded photo, then put the real URL in the README.

## 2b. Streamlit Community Cloud

Create an app from the repository with entrypoint `app.py` and Python 3.12, then add `LENS_EXPERIMENT_URL` under **Advanced settings → Secrets** as an environment variable. TensorFlow is close to the free tier's memory limit; if the app restarts repeatedly, use Hugging Face Spaces instead.

## 2c. Any container host

```bash
docker build -t cifar-lens .
docker run -p 8501:8501 -e LENS_EXPERIMENT_URL=<release zip url> cifar-lens
```

The image runs as a non-root user and has a health check on `/_stcore/health`. Training checkpoints, tests and notebooks are excluded from the image.

Uploads are processed in memory and are not saved by the application.
