from pathlib import Path
import json
import hashlib
import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from lens.data import CLASSES, prepare_image
from lens.metrics import softmax, flip_averaged_logits

ROOT = Path(__file__).parent
st.set_page_config(page_title='CIFAR Lens · Vision Lab', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px;padding-top:2rem} h1{letter-spacing:-.05em;font-size:3.6rem!important}
.hero{padding:2rem 0 1.4rem} .eyebrow{color:#5CE0C6;letter-spacing:.2em;font-size:.78rem;font-weight:700}
.hero p{color:#A8B5CE;font-size:1.15rem;max-width:720px}
[data-testid="stMetric"]{background:#151D32;border:1px solid #28314B;border-radius:14px;padding:18px}
</style><div class="hero"><div class="eyebrow">CIFAR LENS / COMPUTER VISION LAB</div>
<h1>See the prediction.<br>Explore the evidence.</h1>
<p>A hands-on lab for image classification, model confidence, and the mistakes that tell us what to improve.</p></div>''', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading model…')
def load_model(path, modified, expected_sha256):
    """Verify the checksum once per model file version, then keep the model in memory."""
    import tensorflow as tf
    from lens.explain import build_probe
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Model and evaluation record do not match')
    model = tf.keras.models.load_model(path)
    return model, build_probe(model)


@st.cache_resource(show_spinner='Downloading published model…')
def install_published_experiment(url):
    """Hosted demos: fetch release weights once when LENS_EXPERIMENT_URL is set."""
    from lens.fetch import fetch
    target = ROOT / 'artifacts' / 'published'
    if not (target / 'results.json').exists():
        fetch(url, target)
    return str(target)


if os.environ.get('LENS_EXPERIMENT_URL'):
    try:
        install_published_experiment(os.environ['LENS_EXPERIMENT_URL'])
    except Exception as exc:  # the dashboard still opens with historical evidence
        st.error(f'Could not install the published model: {exc}')


historical = json.loads((ROOT / 'evidence/original_results.json').read_text())
def experiment_order(folder):
    record = json.loads((folder / 'results.json').read_text())
    history = record['history']
    best_epoch = int(np.argmin(history['val_loss']))
    return (record['smoke_run'], -history['val_accuracy'][best_epoch], folder.name)


folders = sorted((p.parent for p in (ROOT / 'artifacts').glob('*/results.json')), key=experiment_order)
with st.sidebar:
    st.markdown('### ◉ CIFAR Lens')
    st.caption('Ten classes. One transparent experiment.')
    for progress_path in sorted((ROOT / 'artifacts').glob('*/progress.json')):
        if not (progress_path.parent / 'results.json').exists():
            try:
                progress = json.loads(progress_path.read_text())
                st.caption(f"Training {progress_path.parent.name}: epoch {progress['epoch']} / {progress['epoch_budget']}")
                st.progress(progress['epoch'] / progress['epoch_budget'])
            except (OSError, json.JSONDecodeError):
                pass
    selected = st.selectbox('Experiment', folders, format_func=lambda p: p.name, key='experiment') if folders else None
    threshold = st.slider('Review threshold', 0.1, 0.99, 0.65, 0.01)
    st.caption('Scores below this threshold receive a review flag. High scores do not guarantee correctness.')
    st.markdown('---')
    st.caption('Built from a 2025 group assignment. Extended with AI assistance; original authors credited in the README.')
results = json.loads((selected / 'results.json').read_text()) if selected else None
if results and results['smoke_run']:
    st.warning('Smoke-test experiment: trained on a small subset to verify the software. This is not a performance benchmark.')
cols = st.columns(4)
cols[0].metric('Historical notebook accuracy', f"{historical['accuracy']:.2%}")
ci = results.get('accuracy_ci95') if results else None
cols[1].metric('Fresh test accuracy', f"{results['accuracy']:.2%}" if results else 'Awaiting training',
               help=f"95% bootstrap interval {ci[0]:.2%}–{ci[1]:.2%}" if ci else None)
cols[2].metric('Macro F1', f"{results['macro_f1']:.3f}" if results else '—')
cols[3].metric('Calibration error', f"{results['ece']:.3f}" if results else '—')
st.caption('The historical result reused the test set as validation. Fresh experiments reserve an independent validation split; these scores are not directly comparable.')
predict, evaluate, mistakes, about = st.tabs(['01 / Predict & explain', '02 / Evaluation', '03 / Error explorer', '04 / How it works'])
with predict:
    left, right = st.columns([1, 1.15], gap='large')
    with left:
        st.subheader('Bring an image')
        st.write('Upload one object from the ten CIFAR-10 classes. The model sees a 32 × 32 version.')
        upload = st.file_uploader('JPEG or PNG · up to 8 MB', type=['jpg', 'jpeg', 'png'])
        st.caption('Images stay in this app session and are not saved by the application.')
        samples = sorted((selected / 'samples').glob('*.png')) if selected else []
        example = st.selectbox('Or try a real CIFAR-10 example', [None] + samples,
                    format_func=lambda p: 'Choose an example' if p is None else p.stem.replace('_', ' ')) if samples else None
    with right:
        if selected is None:
            st.info('Train an experiment to unlock real predictions and Grad-CAM. No predictions are simulated.')
            st.code('python -m lens.train --output artifacts/residual')
        elif upload or example:
            try:
                if upload and upload.size > 8 * 1024 * 1024:
                    raise ValueError('Image exceeds 8 MB')
                image = Image.open(upload if upload else example)
                if image.width * image.height > 20_000_000:
                    raise ValueError('Image exceeds 20 megapixels')
                batch = prepare_image(image)
                model_path = selected / 'model.keras'
                model, probe = load_model(str(model_path), model_path.stat().st_mtime_ns, results['model_sha256'])
                def model_logits(images):
                    return model(images, training=False).numpy()
                logits = flip_averaged_logits(model_logits, batch) if results.get('tta') else model_logits(batch)
                probabilities = softmax(logits, results['temperature'])[0]
                index = int(probabilities.argmax())
                st.subheader(CLASSES[index].title())
                st.metric('Top class score', f'{probabilities[index]:.1%}')
                if probabilities[index] < threshold:
                    st.warning('Review suggested: the model has low confidence.')
                st.bar_chart(dict(zip(CLASSES, probabilities)), horizontal=True)
                from lens.explain import gradcam, overlay
                a, b = st.columns(2)
                small = Image.fromarray(np.uint8(batch[0] * 255))
                a.image(small, caption='What the model sees', width=256)
                b.image(overlay(small, gradcam(model, batch, index, probe)), caption='Grad-CAM · original image view', width=256)
                if results.get('tta'):
                    st.caption('Scores average the original and mirrored image. This heatmap explains the original view for the selected class.')
                st.caption("Grad-CAM highlights regions influencing this score; it does not prove the model's reasoning. Uploaded images may differ substantially from CIFAR-10.")
                st.download_button('Download prediction JSON', json.dumps({'class': CLASSES[index],
                    'scores': dict(zip(CLASSES, probabilities.tolist())), 'temperature': results['temperature'],
                    'model_sha256': results['model_sha256'], 'tta': bool(results.get('tta'))}, indent=2), 'prediction.json', 'application/json')
            except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
                st.error(str(exc))
        else:
            st.info('Upload an image to see all ten scores and a visual explanation.')
with evaluate:
    st.subheader('Measure more than accuracy')
    if results:
        import pandas as pd
        import matplotlib.pyplot as plt
        a, b = st.columns(2)
        h = results['history']
        a.line_chart(pd.DataFrame({'train': h['accuracy'], 'validation': h['val_accuracy']}, index=range(1, len(h['accuracy']) + 1)))
        b.line_chart(pd.DataFrame({'train loss': h['loss'], 'validation loss': h['val_loss']}, index=range(1, len(h['loss']) + 1)))
        fig, ax = plt.subplots(figsize=(9, 7))
        matrix = np.array(results['confusion_matrix'])
        plot = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(10), CLASSES)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
        fig.colorbar(plot, ax=ax); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.dataframe(pd.DataFrame({c: results['classification_report'][c] for c in CLASSES}).T)
        st.subheader('Does confidence match correctness?')
        st.scatter_chart(pd.DataFrame(results['reliability']), x='confidence', y='accuracy', size='count')
        st.caption(f"Validation-fitted temperature: {results['temperature']:.3f}. Raw ECE: {results['raw_ece']:.3f}; calibrated ECE: {results['ece']:.3f}. Lower is better; calibration may not improve on every run.")
        st.download_button('Download experiment record', (selected / 'results.json').read_bytes(), 'results.json', 'application/json')
    else:
        st.write('Historical per-class results extracted from your notebook. Fresh evaluation will appear after training.')
        st.dataframe(historical['per_class'], width='stretch')
with mistakes:
    st.subheader('Start with the confident mistakes')
    st.write('Inspect high-confidence errors to understand where a model can mislead you.')
    if selected:
        data = np.load(selected / 'errors.npz', allow_pickle=False)
        category = st.selectbox('True class', ['All'] + CLASSES, key='error_class')
        indices = [i for i, t in enumerate(data['truth']) if category == 'All' or CLASSES[int(t)] == category]
        columns = st.columns(4)
        for slot, i in enumerate(indices[:24]):
            columns[slot % 4].image(data['images'][i],
                caption=f"True: {CLASSES[int(data['truth'][i])]} · Predicted: {CLASSES[int(data['prediction'][i])]} ({data['confidence'][i]:.0%})",
                width='stretch')
        if not indices:
            st.info('No saved errors for this class.')
    else:
        st.info('Error examples are generated from actual test predictions after training.')
with about:
    st.subheader('From assignment to reproducible lab')
    st.markdown('''**Pipeline:** official training set → stratified training/validation split → augmentation → CNN → validation-only calibration → final test evaluation.

Three architectures share one data protocol: a small **baseline** CNN, a compact **residual** CNN, and **ResNet20** (He et al., 2016) trained with pad-crop-flip augmentation, optional Cutout and label smoothing, SGD with warm-up and cosine decay, and optional flip test-time augmentation. Each improvement is measured, not assumed.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

**Limitations:** 32 × 32 images, ten fixed categories, no reliable unknown-object detector, and no claim of production readiness. The review threshold is a user control, not an out-of-distribution guarantee.

**Original authors:** Rory O'Brien, Amir Hamza, MD Suyaib Rahman. New portfolio extension developed with Codex assistance. See the model card and provenance notes for details.''')
