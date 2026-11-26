<p align="center">✦ &nbsp; HUZI9951 &nbsp; ✦</p>

# Superman or Batman

Upload a picture. A small CNN says which hero it is.

The network in `model.py` is a TinyVGG-style stack: three conv blocks, ReLU, max-pool, then a linear head. Input is 128×128, which three pools reduce to 16×16. It is not a pretrained VGG16. Training notes are in `project_notebook.ipynb`. The dataset came from Kaggle. `app.py` is the Streamlit page.

Demo: [batmanorsuperman.streamlit.app](https://batmanorsuperman.streamlit.app)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Weights load from the `model/` folder next to `app.py`.
